import math
from pathlib import Path
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from random import random
from multiprocessing import cpu_count
from ema_pytorch import EMA
import numpy as np
import os

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _triple, _pair, _single

from torchvision import transforms as T, utils

from scipy.stats import truncnorm

from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration
import cv2

from .dataset import KITTIDataset
from .criteria import MaskedL1Loss, MaskedMSELoss
from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.version import __version__
from .visual import colored_depthmap, save_depthmap
from .GuideNet import Basic2d, Conv1x1, BasicBlock, Guide, Basic2dTrans

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] # [Batch_size, 1] * [1, 32] -> [Batch_size, half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # [Batch_size, dim]
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

def divisible_by(numer, denom):
    return (numer % denom) == 0

class ModifiedUNet(Unet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        multi_features = []
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        multi_features.append(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)
            multi_features.append(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x), multi_features

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t) # a = p2_loss_weight with shape of (timesteps,)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps # split 1001 numbers and normlize to [0, 1]
    v_start = torch.tensor(start / tau).sigmoid() # 0.0474
    v_end = torch.tensor(end / tau).sigmoid() # 0.9526
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GuideFusion(nn.Module):
    def __init__(self,
                 in_dim=1,
                 out_dim=1,
                 size = (352, 1216),
                 block=BasicBlock,
                 bc = 16,
                 depth_layers=[2, 2, 2, 2, 2],
                 guide=Guide,
                 norm_layer=nn.BatchNorm2d,
                 weight_ks=3
    ):
        super().__init__()
        in_channels = bc * 2
        self.inplanes = in_channels
        self._norm_layer = norm_layer
        self.size = list(size)

        self.ccm32 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.ccm16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.ccm8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.ccm4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.ccm2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )

        # encode raw depth
        self.conv_lidar = Basic2d(1, bc * 2, norm_layer=None, kernel_size=5, padding=2)
        self.layer1_lidar = self._make_layer(block, in_channels * 2, depth_layers[0], stride=2) # ch = 64
        self.guide1 = guide(in_channels * 2, in_channels * 2, norm_layer, weight_ks)
        self.layer2_lidar = self._make_layer(block, in_channels * 4, depth_layers[1], stride=2) # ch = 128
        self.guide2 = guide(in_channels * 4, in_channels * 4, norm_layer, weight_ks)
        self.layer3_lidar = self._make_layer(block, in_channels * 8, depth_layers[2], stride=2) # ch = 256
        self.guide3 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.layer4_lidar = self._make_layer(block, in_channels * 8, depth_layers[3], stride=2) # ch = 256
        self.guide4 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.layer5_lidar = self._make_layer(block, in_channels * 8, depth_layers[4], stride=2) # ch = 256

        # decoder
        self.layer5d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer4d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer3d = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer2d = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer1d = Basic2dTrans(in_channels * 2, in_channels, norm_layer)

        self.conv = nn.Conv2d(bc * 2, 1, kernel_size=3, stride=1, padding=1)
        self.ref = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)

        self._initialize_weights()
    
    def forward(self, raw, multi_features):
        # reduce channels of multi-features
        f2 = self.ccm2(multi_features[4]) # 64, 1/2
        f4 = self.ccm4(multi_features[3]) # 128, 1/4
        f8 = self.ccm8(multi_features[2]) # 256, 1/8
        f16 = self.ccm16(multi_features[1]) # 256, 1/16
        f32 = self.ccm32(multi_features[0]) # 256, 1/32
        
        # encode raw depth
        c0_lidar = self.conv_lidar(raw) # [32, H, W]
        c1_lidar = self.layer1_lidar(c0_lidar) # 64, H/2, W/2
        c1_lidar_dyn = self.guide1(c1_lidar, f2)
        c2_lidar = self.layer2_lidar(c1_lidar_dyn) # 128, H/4, W/4
        c2_lidar_dyn = self.guide2(c2_lidar, f4)
        c3_lidar = self.layer3_lidar(c2_lidar_dyn) # 256, H/8, W/8
        c3_lidar_dyn = self.guide3(c3_lidar, f8)
        c4_lidar = self.layer4_lidar(c3_lidar_dyn) # 512, H/16, W/16
        c4_lidar_dyn = self.guide4(c4_lidar, f16)
        c5_lidar = self.layer5_lidar(c4_lidar_dyn) # 256, H/32, W/32

        # decoder
        c5 = c5_lidar + f32
        dc5 = self.layer5d(c5)
        c4 = dc5 + c4_lidar_dyn
        dc4 = self.layer4d(c4)
        c3 = dc4 + c3_lidar_dyn
        dc3 = self.layer3d(c3)
        c2 = dc3 + c2_lidar_dyn
        dc2 = self.layer2d(c2)
        c1 = dc2 + c1_lidar_dyn
        dc1 = self.layer1d(c1)
        c0 = dc1 + c0_lidar
        output = self.ref(c0)
        output = self.conv(output)

        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class SpatialDiffusionNet(nn.Module):
    def __init__(
        self, 
        model, 
        image_size, 
        timesteps = 1000, 
        sampling_timesteps = None, 
        loss_type = 'l2', 
        objective = 'pred_noise', 
        beta_schedule = 'sigmoid', 
        schedule_fn_kwargs = dict(), 
        p2_loss_weight_gamma = 0., 
        p2_loss_weight_k = 1, 
        ddim_sampling_eta = 0.,
        auto_normalize = True
    ):
        super().__init__()

        self.model = model

        self.fusion_module = GuideFusion(in_dim=1, out_dim=1, size=image_size, norm_layer=nn.SyncBatchNorm, weight_ks=1)

        self.channels = self.model.channels
        # self.self_condition = self.model.self_condition
        self.self_condition = False

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True)

            imgs.append(img)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            # return F.l1_loss
            # return nn.L1Loss()
            return MaskedL1Loss()
        elif self.loss_type == 'l2':
            # return F.mse_loss
            # return nn.MSELoss(reduction='none')
            return MaskedMSELoss()
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, sparse, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # input = torch.cat((x, interp_d), dim=1)

        _, multi_features = self.model(x, t)
        out = self.fusion_module(sparse, multi_features)

        return out

    def forward(self, img, sparse, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size[0]}*{img_size[1]}'
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        t = torch.tensor([50] * b, dtype=torch.long, device=device)

        img = self.normalize(img)
        return self.p_losses(img, t, sparse, *args, **kwargs)


class DiffDepth(object):
    def __init__(
        self,
        diffusion_model,
        data_folder,
        ckpt_path=None,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        epoch = 100,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        out_size=(352, 1216),
        start_ep = 0
    ):
        super().__init__()

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            mixed_precision = 'fp16' if fp16 else 'no',
            dataloader_config=DataLoaderConfiguration(split_batches=split_batches),
            kwargs_handlers=[ddp_kwargs]
        )

        self.accelerator.native_amp = amp
        self.model = diffusion_model
        self.data_folder = data_folder
        self.batch_size = train_batch_size
        self.augment_horizontal_flip = augment_horizontal_flip
        self.epoch = epoch
        self.start_ep = start_ep
        self.image_size = diffusion_model.image_size
        self.ckpt_path = ckpt_path
        self.out_size= out_size
        
        # setup dataset and dataloader
        self.ds = KITTIDataset(self.data_folder, self.image_size, mode = 'few', augment_horizontal_flip = self.augment_horizontal_flip)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        self.dl = self.accelerator.prepare(dl)
        
        # load checkpoint if exists
        if self.ckpt_path is not None:
            if self.ds.mode == 'train':
                self.load(self.ckpt_path, strict=False, mode='ddpm')
            else:
                self.load(self.ckpt_path, strict=True, mode='ddpm')
            for param in self.model.model.parameters():
                param.requires_grad = False

            model_named_params = [p for _, p in self.model.named_parameters() if p.requires_grad == True]
        
        self.opt = Adam(model_named_params, lr = train_lr, betas = adam_betas, weight_decay=1e-6)
        self.lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', patience=3, verbose=False)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.lr_schedule = self.accelerator.prepare(self.model, self.opt, self.lr_schedule)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            # 'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        return os.path.join(self.results_folder, 'model-{}.pt'.format(milestone))

    def clean_state_dict(self, state_dict):
        new_state_dict = {}
        for key in state_dict.keys():
            if 'model' not in key:
                continue
            else:
                new_key = key.replace('model.', '')
                new_state_dict[new_key] = state_dict[key]

        return new_state_dict

    def load(self, ckpt_path, strict=True, mode='ddpm'):
        accelerator = self.accelerator
        device = accelerator.device

        if mode == 'ddpm':
            data = self.clean_state_dict(torch.load(ckpt_path, map_location=device)['model'])
            model = self.accelerator.unwrap_model(self.model)
            model.model.load_state_dict(data, strict=strict)
        else:
            data = torch.load(ckpt_path, map_location=device)
            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'], strict=strict)

        if 'version' in data:
            print(f"loading from version {data['version']}")
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        criteria = MaskedMSELoss()

        self.model.train()
        for ep in range(self.start_ep, self.start_ep + self.epoch):
            print("=> starting training epoch {} ..".format(ep))

            for idx, sample in enumerate(self.dl):
                depth_loss = 0

                with self.accelerator.autocast():
                    pred = self.model(sample['rgb'].to(device), sample['raw'].to(device))
                    target = sample['gt'].to(device)
                    depth_loss = criteria(pred, target)
                self.accelerator.backward(depth_loss)
                if idx % 10 == 0 and idx != 0 and accelerator.is_local_main_process:
                    print("loss:", depth_loss.item(), " epoch:", ep, " ", idx, "/", len(self.dl))
                # accelerator.clip_grad_norm_(self.model.parameters())
                accelerator.wait_for_everyone()
                
                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                # do validation
                if idx % 500 == 0 and idx != 0:
                    valid_mask = target > 0.1

                    # convert from meters to mm
                    output_mm = 1e3 * pred[valid_mask]
                    target_mm = 1e3 * target[valid_mask]

                    abs_diff = (output_mm - target_mm).abs()

                    mse = float((torch.pow(abs_diff, 2)).mean())
                    rmse = math.sqrt(mse)
                    print('RMSE: {}'.format(rmse))

                    filename = os.path.join(self.results_folder, 'depth_pred_ep{}_idx{}.png'.format(ep, idx))
                    # print(pred[0].shape)
                    save_depthmap(colored_depthmap(pred[0].squeeze().detach().cpu()), filename)
            

            # save the model
            pth_path = self.save(milestone=ep)
    
    def val(self, val_ckpt):
        # load ckpt
        if val_ckpt == None:
            self.load(self.ckpt_path)
        else:
            self.load(val_ckpt, mode='fsdc')
        print("=> Loading test checkpoint...")

        # load validate dataset
        val_ds = KITTIDataset(self.data_folder, self.out_size, mode='val', augment_horizontal_flip=self.augment_horizontal_flip)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=cpu_count())
        print("=> Validation dataset created.")

        # change to eval mode
        device = self.accelerator.device
        self.model.eval()
        print("=> Start evaluating...")
        rmse_all = 0
        mae_all = 0
        irmse_all = 0
        imae_all = 0
        for idx, sample in tqdm(enumerate(val_dl)):
            with torch.no_grad():
                with self.accelerator.autocast():
                    pred = self.model(sample['rgb'].to(device), sample['raw'].to(device))
                    target = sample['gt'].to(device)
                # self.accelerator.wait_for_everyone()

            # calculate metrics
            valid_mask = target > 0.1

            # convert from meters to mm
            output_mm = 1e3 * pred[valid_mask]
            target_mm = 1e3 * target[valid_mask]

            abs_diff = (output_mm - target_mm).abs()
            mae = float(abs_diff.mean())
            mse = float((torch.pow(abs_diff, 2)).mean())
            rmse = math.sqrt(mse)
            rmse_all += rmse
            mae_all += mae
            
            inv_output_km = (1e-3 * pred[valid_mask])**(-1)
            inv_target_km = (1e-3 * target[valid_mask])**(-1)
            abs_inv_diff = (inv_output_km - inv_target_km).abs()
            irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
            imae = float(abs_inv_diff.mean())
            irmse_all += irmse
            imae_all += imae

        rmse_ave = rmse_all / len(val_dl)
        mae_ave = mae_all / len(val_dl)
        irmse_ave = irmse_all / len(val_dl)
        imae_ave = imae_all / len(val_dl)
        print('RMSE: {}'.format(rmse_ave))
        print('MAE: {}'.format(mae_ave))
        print('iRMSE: {}'.format(irmse_ave))
        print('iMAE: {}'.format(imae_ave))

        return rmse_ave