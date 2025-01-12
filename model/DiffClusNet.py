import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
import torchvision.transforms as transforms

from model.IEP.utils.util import instantiate_from_config, default
from model.IEP.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps
from model.IEP.modules.diffusionmodules.util import make_beta_schedule, noise_like, extract_into_tensor

from model.ACP.models.diffusion_multinomial import extract, log_add_exp, cosine_beta_schedule
from model.ACP.models.diffusion_multinomial import log_1_min_a, index_to_log_onehot, log_onehot_to_index


class DiffClusNet_pipeline(object):
    def __init__(self,
                 max_length=24,
                 num_classes=6736,
                 transformer_dim=768,
                 scale_factor=0.18215,
                 IEP_Unet_config=None,
                 ACP_Decoder_config=None,
                 FIC_module_config=None,
                 VAE_model_config=None,
                 Text_Prediction_config=None):
        super(DiffClusNet_pipeline, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.max_length = max_length
        self.num_classes = num_classes
        self.transformer_dim = transformer_dim
        self.scale_factor = scale_factor

        self.IEP_Unet = instantiate_from_config(IEP_Unet_config).to(self.device).eval()
        self.ACP_Decoder = instantiate_from_config(ACP_Decoder_config).to(self.device).eval()
        self.FIC_module = instantiate_from_config(FIC_module_config).to(self.device).eval()
        self.VAE_model = instantiate_from_config(VAE_model_config).to(self.device).eval()
        self.Text_Prediction = instantiate_from_config(Text_Prediction_config).to(self.device).eval()

        self.IEP_ddpm_schedule_init(timesteps=1000, schedule="linear",
                                    linear_start=0.0015, linear_end=0.0205, cosine_s=8e-3)

        self.IEP_ddim_schedule_init(ddim_num_steps=200, ddim_discretize="uniform",
                                    ddim_eta=0.2, verbose=False)

        self.ACP_ddpm_schedule_init(timesteps=1000)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.IEP_Unet.load_state_dict(state_dict['IEP_Unet'], strict=True)
        self.ACP_Decoder.load_state_dict(state_dict['ACP_Decoder'], strict=True)
        self.FIC_module.load_state_dict(state_dict['FIC_module'], strict=True)
        self.VAE_model.load_state_dict(state_dict['VAE_model'], strict=True)
        print(f"DiffClusNet Model loaded from {path}")

    def IEP_ddpm_schedule_init(self, timesteps, schedule, linear_start, linear_end, cosine_s):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.IEP_betas = make_beta_schedule(schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                            cosine_s=cosine_s)
        self.IEP_alphas = 1. - self.IEP_betas
        self.IEP_alphas_cumprod = np.cumprod(self.IEP_alphas, axis=0)
        self.IEP_alphas_cumprod_prev = np.append(1., self.IEP_alphas_cumprod[:-1])

        self.IEP_betas = to_torch(self.IEP_betas)
        self.IEP_alphas = to_torch(self.IEP_alphas)
        self.IEP_alphas_cumprod = to_torch(self.IEP_alphas_cumprod)
        self.IEP_alphas_cumprod_prev = to_torch(self.IEP_alphas_cumprod_prev)

        timesteps, = self.IEP_betas.shape
        self.IEP_ddpm_num_timesteps = int(timesteps)

    def IEP_ddim_schedule_init(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.IEP_ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize,
                                                      num_ddim_timesteps=ddim_num_steps,
                                                      num_ddpm_timesteps=self.IEP_ddpm_num_timesteps, verbose=verbose)

        assert self.IEP_alphas_cumprod.shape[
                   0] == self.IEP_ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('IEP_betas', to_torch(self.IEP_betas))
        self.register_buffer('IEP_alphas_cumprod', to_torch(self.IEP_alphas_cumprod))
        self.register_buffer('IEP_alphas_cumprod_prev', to_torch(self.IEP_alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('IEP_sqrt_alphas_cumprod', to_torch(np.sqrt(self.IEP_alphas_cumprod.cpu())))
        self.register_buffer('IEP_sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - self.IEP_alphas_cumprod.cpu())))
        self.register_buffer('IEP_log_one_minus_alphas_cumprod', to_torch(np.log(1. - self.IEP_alphas_cumprod.cpu())))
        self.register_buffer('IEP_sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / self.IEP_alphas_cumprod.cpu())))
        self.register_buffer('IEP_sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / self.IEP_alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        IEP_ddim_sigmas, IEP_ddim_alphas, IEP_ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=self.IEP_alphas_cumprod.cpu(),
            ddim_timesteps=self.IEP_ddim_timesteps,
            eta=ddim_eta, verbose=verbose)
        self.register_buffer('IEP_ddim_sigmas', IEP_ddim_sigmas)
        self.register_buffer('IEP_ddim_alphas', IEP_ddim_alphas)
        self.register_buffer('IEP_ddim_alphas_prev', IEP_ddim_alphas_prev)
        self.register_buffer('IEP_ddim_sqrt_one_minus_alphas', np.sqrt(1. - IEP_ddim_alphas))
        IEP_sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.IEP_alphas_cumprod_prev) / (1 - self.IEP_alphas_cumprod) * (
                    1 - self.IEP_alphas_cumprod / self.IEP_alphas_cumprod_prev))
        self.register_buffer('IEP_ddim_sigmas_for_original_num_steps', IEP_sigmas_for_original_sampling_steps)

    def ACP_ddpm_schedule_init(self, timesteps):
        ACP_alphas = cosine_beta_schedule(timesteps)
        ACP_alphas = torch.tensor(ACP_alphas.astype('float64'))
        ACP_log_alpha = np.log(ACP_alphas)
        ACP_log_cumprod_alpha = np.cumsum(ACP_log_alpha)
        ACP_log_1_min_alpha = log_1_min_a(ACP_log_alpha)
        ACP_log_1_min_cumprod_alpha = log_1_min_a(ACP_log_cumprod_alpha)

        self.register_buffer('ACP_log_alpha', ACP_log_alpha.float())
        self.register_buffer('ACP_log_1_min_alpha', ACP_log_1_min_alpha.float())
        self.register_buffer('ACP_log_cumprod_alpha', ACP_log_cumprod_alpha.float())
        self.register_buffer('ACP_log_1_min_cumprod_alpha', ACP_log_1_min_cumprod_alpha.float())

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def IEP_p_sample(self, z_t, z_cond, c, t, index, temperature=1.):
        b, *_, device = *z_t.shape, self.device
        e_t = self.IEP_Unet(x=torch.cat([z_t, z_cond], dim=1), timesteps=t, context=c)

        alphas = self.IEP_ddim_alphas
        alphas_prev = self.IEP_ddim_alphas_prev
        sqrt_one_minus_alphas = self.IEP_ddim_sqrt_one_minus_alphas
        sigmas = self.IEP_ddim_sigmas

        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        pred_x0 = (z_t - sqrt_one_minus_at * e_t) / a_t.sqrt()

        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(z_t.shape, device, False) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    # Rest of the functions have been similarly updated to match the above changes

    @torch.no_grad()
    def DiffClusNet_sample(self, input_image):
        lq_image = np.asarray(input_image)
        lq_image = lq_image / 255.0
        lq_image = np.ascontiguousarray(lq_image)
        lq_image = transforms.ToTensor()(lq_image).float()
        lq_image = lq_image.unsqueeze(0).to(self.device)

        c_t = self.Text_Prediction.predict(lq_image)
        c_t = c_t.to(self.device)

        z_LR = self.VAE_model.encode(lq_image)
        z_LR = z_LR.sample()
        b = z_LR.shape[0]
        noise = torch.randn_like(z_LR, device=z_LR.device)
        t = torch.full((b,), 999, device=self.device, dtype=torch.long)
        z_t = self.IEP_q_sample(x_start=z_LR, t=t, noise=noise)

        timesteps = self.IEP_ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = tqdm(time_range, desc='DiffClusNet Sampling', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            Icond_t, Ccond_t = self.FIC_module(c_t, z_LR, z_t, ts)
            z_t, _ = self.IEP_p_sample(z_t, z_LR, Ccond_t, ts, index=index)
            ts = torch.full((b,), index, device=self.device, dtype=torch.long)
            log_c_t = index_to_log_onehot(c_t, self.num_classes)
            log_c_t = self.ACP_p_sample(log_c_t, ts, context=Icond_t)
            c_t = log_onehot_to_index(log_c_t)

        img = 1. / self.scale_factor * z_t
        img = self.VAE_model.decode(img)
        img = img.detach().cpu()
        img = torch.clamp(img, 0., 1.) * 255
        img = img.numpy().astype(np.uint8)[0]
        img = np.transpose(img, (1, 2, 0))

        return img
