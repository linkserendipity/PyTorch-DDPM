#@ ! CK+ datasets 
# #@ test
#!eeg_embedding from DEAP using CCNN to extract feature vectors 
#@ cross attention
import os
import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many
from IPython import embed

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from wechat_push import wx_push
import argparse
from log import logger
from tensorboardX import SummaryWriter
# from torchvision import models
import numpy as np
import random #@ random.randint(0,len(feature_list[i]))
from PIL import Image
import logging

# from torcheeg import transforms as eeg_transforms
# from torcheeg.datasets import DEAPDataset
# from torcheeg.datasets.constants.emotion_recognition.deap import \
#     DEAP_CHANNEL_LOCATION_DICT
# from torcheeg.model_selection import KFoldGroupbyTrial
# from torcheeg.models import CCNN, CCNN_emb
# from torcheeg.trainers import ClassificationTrainer

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

#TODO layerNorm  cross attention 输入先norm 一下啊 输出再norm一下
class LayerNorm(nn.Module):
    def __init__(self, feats, stable=True, dim=-1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim
        if self.stable:
            x = x / x.amax(dim=dim, keepdim=True).detach()
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)
        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)

#* 返回自己？
class Always(): 
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int, optional): controls the minimum frequency of the embeddings. Defaults to 10000.

    Returns:
        Tensor: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, t, y):
        """
        Apply the module to `x` given `t` timestep embeddings, `y` conditional embedding same shape as t.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """

    def forward(self, x, t, y):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t, y)
            else:
                x = layer(x)
        return x


def norm_layer(channels):
    return nn.GroupNorm(32, channels)


class Block(nn.Module):#TODO  这个是用来做啥的？
    def __init__(
        self,
        dim,
        dim_out,
        groups=8,
        norm=True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)

    def forward(self, x, scale_shift=None):
        x = self.groupnorm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift #TODO 这是给 feature map 乘系数加上 time_embedding 的偏置？

        x = self.activation(x)
        return self.project(x)
        #* 先groupnorm然后silu激活然后conv2改变维度


class CrossAttention(nn.Module): #@ crossattention 把 label embedding 和 图像 feature map 交互
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        norm_context=False, #! 这里 False用不到啊
        cosine_sim_attn=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = dim if context_dim is None else context_dim

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x, context):
        b, n, device = *x.shape[:2], x.device
        x = self.norm(x)
        context = self.norm_context(context)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)
        # add null key / value for classifier free guidance in prior net
        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b h 1 d', h=self.heads,  b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        q = q * self.scale
        # similarities
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.cosine_sim_scale
        # masking
        max_neg_value = -torch.finfo(sim.dtype).max
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')
        out = torch.einsum('b i n, b c n -> b c i', context.softmax(dim=-1), x)
        out = rearrange(out, '... -> ... 1')
        return self.net(out)

#TODO 在这里把cross attention GlobalContext加进去
class ResidualBlock(TimestepBlock):
    def __init__(self, dim_in, dim_out, time_dim, dropout, use_global_context=False, groups=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(dim_in),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(), # 这里已经加了激活函数
            nn.Linear(time_dim, dim_out*2) #TODO  从time_dim 降到 dim_out*2  为什么要*2？？？
        )

        self.conv2 = nn.Sequential(
            norm_layer(dim_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        )

        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        if dim_in != dim_out:
            self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        cond_dim = time_dim
        self.gca = GlobalContext(dim_in=dim_out, dim_out=dim_out) if use_global_context else Always(1)
        self.cross_attn = CrossAttention(dim=dim_out, context_dim=cond_dim,)

    def forward(self, x, t, y):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        `y` has shape `[batch_size, time_dim]` #@ 改成 `[batch_size, num_time_tokens, cond_dim]`
        """
        # h = self.conv1(x)
        h = self.block1(x)

        # Add time step embeddings
        # h += self.time_emb(t)[:, :, None, None]
        # h = self.conv2(h)
        context = y #* 把y作为 context 输入
#         print("h.shape", h.shape, "x.shape", x.shape, "context.shape", context.shape, "t.shape", t.shape, "y.shape", y.shape)
        size = h.size(-2)
        hidden = rearrange(h, 'b c h w -> b (h w) c')
        #TODO y=None 怎么处理？ attn=None 不加到h 那里就完事了
        if context == None:
            attn = None
        else:
            attn = self.cross_attn(hidden, context)
            # print('attn = self.cross_attn(hidden, context)')
            # embed()
            # print("attn.shape", attn.shape)
            attn = rearrange(attn, 'b (h w) c -> b c h w', h=size)
            h += attn
        #TODO 这里还有一个问题 time_embedding 不要加上 label_embedding吗？
        t = self.time_emb(t)
        # print('t = self.time_emb(t)')
        # embed()
        t = rearrange(t, 'b c -> b c 1 1')
        scale_shift = t.chunk(2, dim=1) #TODO .chunk 是干嘛的
        h = self.block2(h, scale_shift=scale_shift)
        # print('h = self.block2(h, scale_shift=scale_shift)')
        # embed()
        h *= self.gca(h)
        # print('h *= self.gca(h)')
        # embed()
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        """
        Attention block with shortcut

        Args:
            channels (int): channels
            num_heads (int, optional): attention heads. Defaults to 1.
        """
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding
    """

    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2), #@ 改成 (1,2,2,4) 试试？？
        conv_resample=True,
        num_heads=4,
        label_num=4, #@ sad angry calm/relax happy
        eeg_feature_dim=1024,
        num_time_tokens=2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.eeg_feature_dim = eeg_feature_dim

        # time embedding
        time_embed_dim = model_channels * 4 #NOTE 96*4?? 为什么不是128
        cond_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        #DONE label要改成EEG embedding啦 1024维的向量进来啦
        # if label_num is not None: 
            # self.Label_emb = nn.Embedding(label_num, time_embed_dim) #@
        if label_num is not None: 
            self.Label_emb = nn.Linear(eeg_feature_dim, time_embed_dim)
            #TODO 这里是不是要添加 BN relu()??
            #!!!!!!!!!!!!!!!!
            self.to_time_tokens = nn.Sequential(
            nn.Linear(time_embed_dim, num_time_tokens * cond_dim),
            Rearrange('b (r d) -> b r d', r=num_time_tokens) #* (batch 2*time_embed_dim) -> (batch 2 time_embed_dim)
            )



        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions: #TODO 这个ds 是什么  attention_resolutions 是什么
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers)) #@ 添加
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, y):
        """Apply the model to an input batch.

        Args:
            x (Tensor): [N x C x H x W]
            timesteps (Tensor):[N,] a 1-D batch of timesteps.
            y (Tensor): [N,] LongTensor conditional labels.

        Returns:
            Tensor: [N x C x ...]
        """
        
        # time step embedding #* emb.shape=torch.Size([8, 384]) (batch_size, model_channels*4)
        #TODO Unet forward() 修改 embedding 方式
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if y is not None: 
            # t_emb += self.Label_emb(y) #TODO t + y nn.Embedding把7个数字映射到 384维向量了 所以要把nn.Embedding改成nn.Linear??
            y = self.Label_emb(y) #* y.shape=torch.Size([12, 384])
            # embed()
            y = self.to_time_tokens(y) #* y.shape=torch.Size([12, 2, 384]) 

        # embed() #@ time embedding
        hs = []
        # down stage
        h = x #* torch.Size([8, 1, 128, 128]) (batch_size, in_channels, images_size, images_size)
        for module in self.down_blocks:
            h = module(h, t_emb, y)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, t_emb, y)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t_emb, y) #@ h.shape=torch.Size([8, 96, 128, 128])
        # embed()
        return self.out(h)

def linear_beta_schedule(timesteps):
    """
    beta schedule
    """
    scale = 1000 / timesteps #! NOTE timesteps=5000 scale=0.2 beta_end=0.04 timesteps设为1000？
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        # beta_schedule='linear',
        beta_schedule='cosine', #NOTE 这里改了
        CFG_scale=1.5
    ):
        self.timesteps = timesteps
        self.CFG_scale = CFG_scale #@@@

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, a, t, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start, t):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t, t, y, clip_denoised=True):
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        cfg_scale=self.CFG_scale #@@self. 这样写就可以在函数里调用Class的self.定义的变量
        pred_noise = model(x_t, t, y)
        if cfg_scale>0:
            uncond_pred_noise = model(x_t, t, None) #@ unconditional predict noise
            #TODO 这里生成的时候 y=None 考虑进去
            pred_noise = torch.lerp(uncond_pred_noise, pred_noise, cfg_scale) #@ out = start + weight * (end - start). torch.lerp(start, end, weight, out=None)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    #@ 作sample的函数不改变梯度
    @torch.no_grad()
    def p_sample(self, model, x_t, t, y, clip_denoised=True):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, y, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def p_sample_loop(self, model, y, shape):
        # denoise: reverse diffusion
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, y)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, y, image_size, batch_size=8, channels=3):
        # sample new images
        return self.p_sample_loop(model, y, shape=(batch_size, channels, image_size, image_size))

    def train_losses(self, model, x_start, t, y): #TODO 这里的y 从 labels 变成 label_emb啦！ torch.Size([24]) -> torch.Size([24, 1024])
        # compute train losses
        # generate random noise
        noise = torch.randn_like(x_start)
        #* get x_t x_t=sqrt(alpha_bar_t)*x_start+sqrt(1-alpha_bar_t)*z
        x_noisy = self.q_sample(x_start, t, noise=noise)
        if np.random.random() < 0.1: #@@@@ 10% 随机drop掉 label为None Unconditional训练
            y=None #TODO 随机数被固定了怎么办
        predicted_noise = model(x_noisy, t, y)
        loss = F.mse_loss(noise, predicted_noise)
        return loss


parser = argparse.ArgumentParser()

# parser.add_argument('--dir', type=str, required=True, help="data dir")
parser.add_argument('--dir', type=str, default='/media/SSD/lingsen/data/CK+/results/VA', help="data dir")
parser.add_argument('--eeg_dir', type=str, default='/media/SSD/lingsen/data/EEG/DEAP/data_preprocessed_python', help="EEG data dir")
parser.add_argument('--save_dir', type=str, default='/media/SSD/lingsen/code/PyTorch-DDPM/save_model/CFG_y_model.pth', help="save model dir")
parser.add_argument('--eeg_save_dir', type=str, default='/media/SSD/lingsen/code/EEG-Conformer/tmp_out/deap_ccnn_va/weight', help="eeg model dir")
parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
parser.add_argument('--eeg_batch_size', type=int, default=256, help="EEG CCNN batch_size")
parser.add_argument('--timesteps', type=int, default=5000, help="timesteps")
parser.add_argument('--epochs', type=int, default=200, help="epochs")
# parser.add_argument('--image_size', type=int, default=64, help="image_size")
parser.add_argument('--image_size', type=int, default=128, help="image_size")
parser.add_argument('--gpuid', type=int, default=3, help="GPU ID")
parser.add_argument('--scale', type=float, default=1.5, help="CFG_scale")
parser.add_argument('--label', type=int, default=0, help="generated emotional images")
parser.add_argument('--gn', type=int, default=16, help="generated image number")
parser.add_argument('--ge', type=int, default=100, help="generated epochs for i")

args = parser.parse_args()
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ main 开始训练

data_dir = args.dir 
save_dir = args.save_dir
batch_size = args.batch_size
timesteps = args.timesteps
image_size = args.image_size
epochs = args.epochs
gpuid = args.gpuid
scale = args.scale
label = args.label
generate_number = args.gn
generate_epochs = args.ge

eeg_dir = args.eeg_dir
eeg_save_dir = args.eeg_save_dir
eeg_batch_size= args.eeg_batch_size
logger.info("Begin!")
start_time = round(time.monotonic()) # *
###############################################################################
# Pre-experiment Preparation to Ensure Reproducibility
# -----------------------------------------
# Use the logging module to store output in a log file for easy reference while printing it to the screen.
'''

os.makedirs('./tmp_out/deap_ccnn_va_eeg/log', exist_ok=True)
logger = logging.getLogger('using CCNN model to extract EEG feature from the DEAP Dataset')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./tmp_out/deap_ccnn_va_eeg/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

###############################################################################
# Set the random number seed in all modules to guarantee the same result when running again.
#TODO 这个设置了之后 下一次运行代码 每次随机生成的数是不一样的，但是和上一次对应的位置的数是一样的 所以得到diffusion里面取消固定的seed? random.seed(None) np.random.seed(None) ??????????????????
def seed_everything(seed):
    random.seed(seed) #@ 这个为什么也要固定？？
    np.random.seed(seed) #TODO 这个要不得吧？
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #TODO 为什么是False

seed_everything(42)
###############################################################################
# Customize Trainer
# -----------------------------------------
# TorchEEG provides a large number of trainers to help complete the training of classification models, generative models and cross-domain methods. Here we choose the simplest classification trainer, inherit the trainer and overload the log function to save the log using our own defined method; other hook functions can also be overloaded to meet special needs.
class MyClassificationTrainer(ClassificationTrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)


# dataset = DEAPDataset(io_path=f'./tmp_out/deap_ccnn_va/deap',
dataset = DEAPDataset(io_path=f'/media/SSD/lingsen/code/EEG-Conformer/tmp_out/deap_ccnn_va/deap',
                      root_path=eeg_dir, #@ eeg_data dir 
                      offline_transform=eeg_transforms.Compose([ #@  Band DE 
                          eeg_transforms.BandDifferentialEntropy( 
                              sampling_rate=128, apply_to_baseline=True),
                          eeg_transforms.BaselineRemoval(),
                          eeg_transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT) #@ 转成9*9的Grid
                      ]),
                      online_transform=eeg_transforms.ToTensor(),
                      label_transform=eeg_transforms.Compose([ #@ valence 二分类
                          eeg_transforms.Select(['valence', 'arousal']), #@ list形式输入多个 value
                          eeg_transforms.Binary(5.0), 
                          eeg_transforms.BinariesToCategory() #@ 添加多标签 [0, 1] 转成0, 1, 2, 3
                      ]),
                      chunk_size=128,
                      baseline_chunk_size=128,
                      num_baseline=3,
                      num_worker=4)

#########################################################
# Step 2: Divide the Training and Test samples in the Dataset
# Here, the dataset is divided using 5-fold cross-validation. In the process of division, we group according to the trial index, and every trial takes 4 folds as training samples and 1 fold as test samples. Samples across trials are aggregated to obtain training set and test set.
k_fold = KFoldGroupbyTrial(n_splits=5,
                           split_path='/media/SSD/lingsen/code/EEG-Conformer/tmp_out/deap_ccnn_va/split') # 用EEG 文件夹下面分好的

######################################################################
# Step 3: Define the Model and Start Training
# We first use a loop to get the dataset in each cross-validation. In each cross-validation, we initialize the CCNN model and define the hyperparameters. For example, each EEG sample contains 4-channel features from 4 sub-bands, the grid size is 9 times 9, etc.
# We then initialize the trainer and set the hyperparameters in the trained model, such as the learning rate, the equipment used, etc. The :obj:`fit` method receives the training dataset and starts training the model. The :obj:`test` method receives a test dataset and reports the test results. The :obj:`save_state_dict` method can save the trained model.

for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
    # Initialize the model
    if i != 3:
        print(f'i={i}')
        continue
    print(f'i=3? {i}')
    # loaded_model = CCNN(in_channels=4, grid_size=(9, 9), num_classes=4, dropout=0.5) #@
    loaded_model = CCNN_emb(in_channels=4, grid_size=(9, 9), num_classes=4, dropout=0.5) #@
    eeg_save_dir_i = os.path.join(eeg_save_dir, f'{i}.pth')
    # save_dir_i = os.path.join(save_dir, f'epoch_80_{i}.pth')
    print(eeg_save_dir_i)
    # /media/SSD/lingsen/code/EEG-Conformer/tmp_out/deap_ccnn_va/weight/epoch_80_0.pth
    # loaded_model.load_state_dict(torch.load(save_dir_i).keys())
    #@ torch.load(save_dir_i).keys() = dict_keys(['model'])
    #@ torch.load(save_dir_i)['model'].keys()=odict_keys(['conv1.1.weight', 'conv1.1.bias', 'conv2.1.weight', 'conv2.1.bias', 'conv3.1.weight', 'conv3.1.bias', 'conv4.1.weight', 'conv4.1.bias', 'lin1.0.weight', 'lin1.0.bias', 'lin2.weight', 'lin2.bias'])
    #? x['model']['conv1.1.weight'].shape = torch.Size([64, 4, 4, 4])
    # Initialize the trainer and use the 0-th GPU for training, or set device_ids=[] to use CPU
    trainer = MyClassificationTrainer(model=loaded_model,
                                      lr=1e-4,
                                      weight_decay=1e-4,
                                      device_ids=[gpuid])
    trainer.load_state_dict(load_path=eeg_save_dir_i)

    # Initialize several batches of training samples and test samples
    train_loader = DataLoader(train_dataset,
                              batch_size=eeg_batch_size,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset,
                            batch_size=eeg_batch_size,
                            shuffle=False,
                            num_workers=4)
    print(f'len(train_loader)={len(train_loader)}') # 240
    print(f'len(val_loader)={len(val_loader)}') # 60
    # Do 50 rounds of training
    # trainer.fit(train_loader, val_loader, num_epochs=epochs)
    # trainer.test(val_loader)
    # trainer.feature_vector(val_loader)

    feature_vector, feature_label = trainer.feature_vector(train_loader) #! train_loader 不是 train_dataset!!!!
    print(feature_vector.shape) #@ torch.Size([15360, 1024]) 61440, 1024
    print(feature_label.shape) #@ torch.Size([15360]) 61440

    index0 = torch.where(feature_label==0)
    index1 = torch.where(feature_label==1)
    index2 = torch.where(feature_label==2)
    index3 = torch.where(feature_label==3)
    feature_vector0 = feature_vector[index0]
    feature_vector1 = feature_vector[index1]
    feature_vector2 = feature_vector[index2]
    feature_vector3 = feature_vector[index3]
    #TODO 不同的数量不能把这四组张量组合成一个大的张量？ 到diffusion里面label对应的feature_vector提出哪一个？
    #! 添加到list里面不就行了 feature_list[0]

    # print(feature_list[0].shape)
    # print(feature_list[1].shape)
    # print(feature_list[2].shape)
    # print(feature_list[3].shape)
    #@
    # torch.Size([12480, 1024]) 
    # torch.Size([14208, 1024])
    # torch.Size([12768, 1024])
    # torch.Size([21984, 1024])
    feature_vector0_npy = feature_vector0.data.cpu().numpy() # 数据类型转换
    np.save("feature_vector0_npy.npy",feature_vector0_npy) # 大功告成

    feature_vector0_npy = np.load("feature_vector0_npy.npy")
    feature_vector0 = torch.from_numpy(feature_vector0_npy)
    feature_list=[feature_vector0, feature_vector1, feature_vector2, feature_vector3]

'''
feature_vector0_npy = np.load("feature_vector0_npy.npy")
feature_vector0 = torch.from_numpy(feature_vector0_npy)
feature_vector1_npy = np.load("feature_vector1_npy.npy")
feature_vector1 = torch.from_numpy(feature_vector1_npy)
feature_vector2_npy = np.load("feature_vector2_npy.npy")
feature_vector2 = torch.from_numpy(feature_vector2_npy)
feature_vector3_npy = np.load("feature_vector3_npy.npy")
feature_vector3 = torch.from_numpy(feature_vector3_npy)
#@ feature_list 后面用于把label 换成 随机采样 label对应的 eeg feature 
feature_list=[feature_vector0, feature_vector1, feature_vector2, feature_vector3]
#TODO 提取 DEAP 训练集和测试集的 feature 分别保存到npy里面去 diffusion训练和测试的时候分别读取对应的feature npy 到tensor
test_feature_vector0_npy = np.load("test_feature_vector0_npy.npy")
test_feature_vector0 = torch.from_numpy(test_feature_vector0_npy)
test_feature_vector1_npy = np.load("test_feature_vector1_npy.npy")
test_feature_vector1 = torch.from_numpy(test_feature_vector1_npy)
test_feature_vector2_npy = np.load("test_feature_vector2_npy.npy")
test_feature_vector2 = torch.from_numpy(test_feature_vector2_npy)
test_feature_vector3_npy = np.load("test_feature_vector3_npy.npy")
test_feature_vector3 = torch.from_numpy(test_feature_vector3_npy)
#@ feature_list 后面用于把label 换成 随机采样 label对应的 eeg feature 
test_feature_list=[test_feature_vector0, test_feature_vector1, test_feature_vector2, test_feature_vector3]
# print(feature_list[0].shape) 
# print(feature_list[1].shape)
# print(feature_list[2].shape)
# print(feature_list[3].shape)

logger.info(" Cross attention ! EEG test feature npy load done! Begin image generation!")
# embed()


transform = transforms.Compose([
    transforms.Grayscale(1), #@ 3通道转成单通道
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
dataset = ImageFolder(data_dir, transform = transform) #!！!!
print("dataset.class_to_idx: ", dataset.class_to_idx)
#TODO 四类情绪图片 0->Sad 1->Angry 2->Calm 3->Happy
#@ LVLA=0 LVHA=1 HVLA=2 HVHA=3
#@ Sad    Angry  Calm?? Happy Calm 就是把Happy的前5张图提出来 Happy是把Happy的后10张图提出来 可能会有重合的部分但是无所谓 

print(dataset.classes)  #根据分的文件夹的名字来确定的类别
# class_1 = dataset.classes[0]
# print(dataset.class_to_idx) #@! 按顺序为这些类别定义索引为0,1...
# print(dataset.imgs) #返回从所有文件夹中得到的图片的路径以及其类别
# class_1 = os.path.split(data_dir)[-1]
# print("class_1:{}".format(class_1) )

message0 = 'eeg_CFG_emb_ca: batch_size={} timesteps={} image_size={} epochs={}'.format(batch_size, timesteps, image_size, epochs)
logger.info(message0)
# ms1=f'gpuid={gpuid}, scale={scale} '
ms1= f'gpuid={gpuid}, scale={scale}, label={label}, generate_number={generate_number}, generate_epochs={generate_epochs}'
logger.info(ms1)
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # use MNIST dataset
# dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#* define model and diffusion
device = "cuda:{}".format(gpuid) if torch.cuda.is_available() else "cpu"

loaded_model = UNetModel(
    in_channels=1, #!!@@@@@@@@@@@@@@@@@@
    model_channels=96, #TODO model_channels=128?
    out_channels=1,
    channel_mult=(1, 2, 2), #TODO  channel_mult=(1, 2, 2, 4) ？
    attention_resolutions=[],
    label_num=len(dataset.classes) #!!!!!!! 变成4类了
)
loaded_model.load_state_dict(torch.load(save_dir, map_location=device))
loaded_model.to(device)
loaded_model.eval()
# y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]).to(device) 
y_o = label*torch.ones([generate_number]).long()
print(f'y_o:{y_o}')
#TODO 改成 test feature生成的文件夹
generate_dir = '/media/SSD/lingsen/data/CK+/results/generated_va_ca_test/{}_{}_{}_{}_{}/{}'.format(image_size, batch_size, epochs, timesteps, scale, label)
print(f'generate_dir: {generate_dir}')
os.makedirs(generate_dir, exist_ok=True) 

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule = 'cosine', CFG_scale=scale) #TODO 改成'cosine'
#TODO test feature
# y = [test_feature_list[y[i].item()][random.randint(0, len(test_feature_list[y[i].item()]))] for i in range(len(y))] #@ 这还是一个list
for i in range(130, generate_epochs): #TODO 101
    print(f'i={i}, generate_epochs={generate_epochs}')
    # embed()
    # y = [feature_list[y_o[i].item()][random.randint(0, len(feature_list[y_o[i].item()])-1)] for i in range(len(y_o))] #@ 这还是一个list
    y = [test_feature_list[y_o[i].item()][random.randint(0, len(test_feature_list[y_o[i].item()])-1)] for i in range(len(y_o))] 
    #TODO 变成 test feature了 估计更多不准的图
    y = torch.stack(y, 0) #@@@@ torch.Size([8, 1024])
    y = y.to(device) # 先别放进device 直接是原始的 list 数字进来

    generated_images = gaussian_diffusion.sample(loaded_model, y, image_size, batch_size=generate_number, channels=1) 

    imgs = generated_images[-1]
    for j in range(generate_number):
        save_image = os.path.join(generate_dir, f'{i*generate_number+j+1}.png')
        image_array = (imgs[j,0]+1.0) * 255 / 2
        im = Image.fromarray(image_array) 
        im = im.convert('L') # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’ 
        im.save(save_image)
    del generated_images


end_time = round(time.monotonic())
# print('Total running time: {}'.format(timedelta(seconds=end_time - start_time))) # ! 打印的时候保留2位小数
logger.info('Total running time: {}'.format(timedelta(seconds=end_time - start_time)))

message1 = 'EEG_emb_CFG_test scale={} batch_size={} timesteps={} image_size={} epochs={} testing time={}'.format(scale, batch_size, timesteps, image_size, epochs, timedelta(seconds=end_time - start_time))

logger.info(message1)
# wx_push(message1) 