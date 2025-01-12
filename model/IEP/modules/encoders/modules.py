import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from model.FIC import FIC_module
from model.IEP.modules.x_transformer import Encoder, TransformerWrapper


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', fic_module=None):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.fic_module = fic_module

    def forward(self, batch, key=None, context=None):
        if key is None:
            key = self.key
        c = batch[key][:, None]
        c = self.embedding(c)
        if context is not None and self.fic_module is not None:
            c = self.fic_module(c, context)  # 融合分类特征
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda", fic_module=None):
        super().__init__()
        self.device = device
        self.fic_module = fic_module
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens, context=None):
        tokens = tokens.to(self.device)
        z = self.transformer(tokens, return_embeddings=True)
        if context is not None and self.fic_module is not None:
            z = self.fic_module(z, context)  # 融合分类路径信息
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizer model and add some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda", use_tokenizer=True, embedding_dropout=0.0, fic_module=None):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        self.fic_module = fic_module
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text, context=None):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        if context is not None and self.fic_module is not None:
            z = self.fic_module(z, context)  # 融合分类路径信息
        return z

    def encode(self, text):
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x, context=None):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        if context is not None:
            x = x + context  # 简单融合
        return x

    def encode(self, x):
        return self(x)


class FeatureMapEncoder(nn.Module):
    def __init__(self, input_dim, n_embed):
        super().__init__()
        self.layer1_conv2 = nn.Conv2d(input_dim, n_embed, 1, 1, 0, groups=32)
        self.layer1_conv2_bn = nn.BatchNorm2d(n_embed)
        self.layer1_conv2_relu = nn.ReLU(inplace=True)

    def forward(self, fea_map, context=None):
        x = self.layer1_conv2(fea_map)
        x = self.layer1_conv2_bn(x)
        x = self.layer1_conv2_relu(x)

        if context is not None:
            x = x + context.unsqueeze(-1).unsqueeze(-1)  # 融合分类特征

        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()

        return x

    def encode(self, fea_map):
        return self(fea_map)