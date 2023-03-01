import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from torch.nn.modules.normalization import GroupNorm
from einops import rearrange
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

#TODO layerNorm  cross attention 输入先norm  输出再norm
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
            nn.SiLU(), # 已经加了激活函数
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
        context = y 
#         print("h.shape", h.shape, "x.shape", x.shape, "context.shape", context.shape, "t.shape", t.shape, "y.shape", y.shape)
        size = h.size(-2)
        hidden = rearrange(h, 'b c h w -> b (h w) c')
        #TODO y=None 怎么处理？ attn=None 不加到h 
        if context == None:
            attn = None
        else:
            attn = self.cross_attn(hidden, context)
            # print('attn = self.cross_attn(hidden, context)')
            # embed()
            # print("attn.shape", attn.shape)
            attn = rearrange(attn, 'b (h w) c -> b c h w', h=size)
            h += attn
        #TODO time_embedding 不要加上 label_embedding？
        t = self.time_emb(t)

        t = rearrange(t, 'b c -> b c 1 1')
        scale_shift = t.chunk(2, dim=1) # .chunk?
        h = self.block2(h, scale_shift=scale_shift)

        h *= self.gca(h)

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
        channel_mult=(1, 2, 2, 2), #@ 改成 (1,2,2,4) ？
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
        time_embed_dim = model_channels * 4 
        cond_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        #DONE label要改成EEG embedding 1024维的向量
        # if label_num is not None: 
            # self.Label_emb = nn.Embedding(label_num, time_embed_dim) #@
        if label_num is not None: 
            # self.Label_emb = nn.Linear(eeg_feature_dim, time_embed_dim)
            self.Label_emb = nn.Sequential(
                nn.SiLU(), # 已经加了激活函数
                nn.Linear(eeg_feature_dim, time_embed_dim) # 从time_dim 降到 dim_out*2  为什么*2
            )
            #TODO 这里是不是要添加 BN relu()??

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
                if ds in attention_resolutions: #TODO ds attention_resolutions
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers)) #@
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
            # t_emb += self.Label_emb(y) 
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