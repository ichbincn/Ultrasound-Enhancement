import torch
import torch.nn as nn
import torch.nn.functional as F
from model.IEP.utils.util import instantiate_from_config


class FIC_model(nn.Module):
    def __init__(self,
                 FIC_Unet=None,
                 FIC_Transformer=None,
                 scale_factor=0.18215,
                 num_classes=6736,
                 context_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.first_embeder = nn.Embedding(num_classes, context_dim)
        self.Unet = instantiate_from_config(MoM_Unet)
        self.Transformer = instantiate_from_config(MoM_Transformer)
        self.scale_factor = scale_factor

        # 添加特征融合层
        self.attn_fusion = nn.MultiheadAttention(embed_dim=context_dim, num_heads=4)

    def forward(self, c_t, z_lr=None, z_t=None, t=None):
        # 对齐 z_lr 和 z_t 的空间维度
        if z_lr.shape[-1] != z_t.shape[-1] or z_lr.shape[-2] != z_t.shape[-2]:
            z_lr = F.interpolate(z_lr, size=z_t.shape[-2:])

        # 将 z_lr 和 z_t 拼接，并按 scale_factor 进行缩放
        z_input = torch.cat((z_lr, z_t), dim=1)
        z_input *= self.scale_factor

        # 对分类条件进行嵌入
        c_t_embed = self.first_embeder(c_t)

        # 使用 Unet 提取图像生成的条件特征
        I_cond = self.Unet(z_input, timesteps=t, context=c_t_embed)

        # 全局特征用于 Transformer 的额外输入
        z_global = torch.mean(z_input, dim=(2, 3))  # 池化提取全局特征
        c_t_with_global = torch.cat((c_t_embed, z_global), dim=1)

        # 使用 Transformer 提取分类条件特征
        C_cond = self.Transformer(c_t_with_global)

        # 特征融合（使用注意力机制）
        I_cond_fused, _ = self.attn_fusion(I_cond.flatten(2).permute(2, 0, 1),
                                           C_cond.unsqueeze(0),
                                           C_cond.unsqueeze(0))
        I_cond_fused = I_cond_fused.permute(1, 2, 0).view(I_cond.size())

        return I_cond_fused, C_cond
