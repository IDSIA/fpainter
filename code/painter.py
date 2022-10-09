# Originally forked from https://github.com/lucidrains/lightweight-gan/blob/main/lightweight_gan/lightweight_gan.py
# Modified to implement FPA models
# The models used in the paper are:
# base: LightGAN baseline
# lstm_delta: FPA with v1 input generator
# lstm_delta_unet: U-Net extension of the model above
# lstm_delta_split_v2: FPA with v2 input generator
# lstm_delta_split_v2_unet: U-Net extension of the model above

import os
import json
import multiprocessing
from random import random
import math
from math import log2, floor
from functools import lru_cache, partial
from contextlib import contextmanager, ExitStack
from pathlib import Path
from shutil import rmtree

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
import torchvision
from torchvision import transforms
from kornia.filters import filter2d

from diff_augment import DiffAugment
from version import __version__

from tqdm import tqdm
from einops import rearrange, reduce, repeat

from lightweight_gan import LightweightGAN
from unet import Unet

from model_utils import TransformerFFlayers, Attention, BaseModel, RFWPlayer

# asserts

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png', 'tiff']

# helpers

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def is_power_of_two(val):
    return log2(val).is_integer()

def default(val, d):
    return val if exists(val) else d

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

def safe_div(n, d):
    try:
        res = n / d
    except ZeroDivisionError:
        prefix = '' if int(n >= 0) else '-'
        res = float(f'{prefix}inf')
    return res

# loss functions

def gen_ce_loss(fake, real):
    return fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

def hinge_loss_bis(real, fake):
    return (F.relu(1 + real).sum() + F.relu(1 - fake).sum()) / (real.shape[0] + fake.shape[0])

def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

@lru_cache(maxsize=10)
def det_randn(*args):
    """
    deterministic random to track the same latent vars (and images) across training steps
    helps to visualize same image over training steps
    """
    return torch.randn(*args)

def interpolate_between(a, b, *, num_samples, dim):
    assert num_samples > 2
    samples = []
    step_size = 0
    for _ in range(num_samples):
        sample = torch.lerp(a, b, step_size)
        samples.append(sample)
        step_size += 1 / (num_samples - 1)
    return torch.stack(samples, dim=dim)

# helper classes

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)
    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)

class Noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise = None):
        b, _, h, w, device = *x.shape, x.device

        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device = device)

        return x + self.weight * noise

def Conv2dSame(dim_in, dim_out, kernel_size, bias = True):
    pad_left = kernel_size // 2
    pad_right = (pad_left - 1) if (kernel_size % 2) == 0 else pad_left

    return nn.Sequential(
        nn.ZeroPad2d((pad_left, pad_right, pad_left, pad_right)),
        nn.Conv2d(dim_in, dim_out, kernel_size, bias = bias)
    )

# attention

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, kernel_size = 3):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.kernel_size = kernel_size
        self.nonlin = nn.GELU()

        self.to_lin_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_lin_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)

        self.to_out = nn.Conv2d(inner_dim * 2, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        # linear attention

        lin_q, lin_k, lin_v = (self.to_lin_q(fmap), *self.to_lin_kv(fmap).chunk(2, dim = 1))
        lin_q, lin_k, lin_v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (lin_q, lin_k, lin_v))

        lin_q = lin_q.softmax(dim = -1)
        lin_k = lin_k.softmax(dim = -2)

        lin_q = lin_q * self.scale

        context = einsum('b n d, b n e -> b d e', lin_k, lin_v)
        lin_out = einsum('b n d, b d e -> b n e', lin_q, context)
        lin_out = rearrange(lin_out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # conv-like full attention

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c x y', h = h), (q, k, v))

        k = F.unfold(k, kernel_size = self.kernel_size, padding = self.kernel_size // 2)
        v = F.unfold(v, kernel_size = self.kernel_size, padding = self.kernel_size // 2)

        k, v = map(lambda t: rearrange(t, 'b (d j) n -> b n j d', d = self.dim_head), (k, v))

        q = rearrange(q, 'b c ... -> b (...) c') * self.scale

        sim = einsum('b i d, b i j d -> b i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        attn = sim.softmax(dim = -1)

        full_out = einsum('b i j, b i j d -> b i d', attn, v)
        full_out = rearrange(full_out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # add outputs of linear attention + conv like full attention

        lin_out = self.nonlin(lin_out)
        out = torch.cat((lin_out, full_out), dim = 1)
        return self.to_out(out)

# dataset

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class identity(object):
    def __call__(self, tensor):
        return tensor

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        transparent = False,
        greyscale = False,
        skip_resize = False,
        transform_to_greyscale = False,
        aug_prob = 0.
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        if transform_to_greyscale:
            assert not greyscale, "Images are already in greyscale."

        if transparent:
            num_channels = 4
            pillow_mode = 'RGBA'
            expand_fn = expand_greyscale(transparent)
        elif greyscale:
            num_channels = 1
            pillow_mode = 'L'
            expand_fn = identity()
        else:
            if transform_to_greyscale:
                num_channels = 1
                pillow_mode = 'L'
                expand_fn = identity()
            else:
                num_channels = 3
                pillow_mode = 'RGB'
                expand_fn = expand_greyscale(transparent)

        convert_image_fn = partial(convert_image_to, pillow_mode)

        if skip_resize:
            self.transform = transforms.Compose([
                transforms.Lambda(convert_image_fn),
                transforms.RandomHorizontalFlip(aug_prob),
                RandomApply(
                    aug_prob,
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                    transforms.CenterCrop(image_size)),
                transforms.ToTensor(),
                transforms.Lambda(expand_fn)
            ])
        elif transform_to_greyscale:
            self.transform = transforms.Compose([
                transforms.Lambda(convert_image_fn),
                transforms.Lambda(partial(resize_to_minimum_size, image_size)),
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomHorizontalFlip(aug_prob),
                RandomApply(
                    aug_prob,
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                    transforms.CenterCrop(image_size)),
                transforms.ToTensor(),
                transforms.Lambda(expand_fn)
            ])   
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(convert_image_fn),
                transforms.Lambda(partial(resize_to_minimum_size, image_size)),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(aug_prob),
                RandomApply(
                    aug_prob,
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                    transforms.CenterCrop(image_size)),
                transforms.ToTensor(),
                transforms.Lambda(expand_fn)
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# augmentations

def random_hflip(tensor, prob):
    if prob < random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False, **kwargs):
        context = torch.no_grad if detach else null_context

        with context():
            if random() < prob:
                images = random_hflip(images, prob=0.5)
                images = DiffAugment(images, types=types)

        return self.D(images, **kwargs)

# modifiable global variables

norm_class = nn.BatchNorm2d

def upsample(scale_factor = 2):
    return nn.Upsample(scale_factor = scale_factor)

# global context network
# https://arxiv.org/abs/2012.13375
# similar to squeeze-excite, but with a simplified attention pooling
# and a subsequent layer norm

class GlobalContext(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim = -1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)

# frequency channel attention
# https://arxiv.org/abs/2012.11879

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    return result * (1 if freq == 0 else math.sqrt(2))

def get_dct_weights(width, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, width)
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for x in range(width):
            for y in range(width):
                coor_value = get_1d_dct(x, u_x, width) * get_1d_dct(y, v_y, width)
                dct_weights[:, i * c_part: (i + 1) * c_part, x, y] = coor_value

    return dct_weights

class FCANet(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out,
        reduction = 4,
        width
    ):
        super().__init__()

        freq_w, freq_h = ([0] * 8), list(range(8))
        dct_weights = get_dct_weights(
            width, chan_in, [*freq_w, *freq_h], [*freq_h, *freq_w])
        self.register_buffer('dct_weights', dct_weights)

        chan_intermediate = max(3, chan_out // reduction)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = reduce(
            x * self.dct_weights,
            'b c (h h1) (w w1) -> b c h1 w1', 'sum', h1 = 1, w1 = 1)
        return self.net(x)


###############################################################################


# Purely additive learning rule
class LSTMSumGenerator(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 256,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers
        self.use_softmax = use_softmax

        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(
                    torch.randn(self.num_decoding_steps, latent_dim)))

        # map latent to LSTM initial states
        self.latent_to_init = nn.Linear(
            latent_dim, 2 * rnn_hidden_size * num_layers)

        self.decoder_rnn = nn.LSTM(latent_dim, rnn_hidden_size, num_layers)
        # channels * 2 for key/value
        self.kv_net = nn.Linear(
            rnn_hidden_size, init_channel * 2 * self.im_size)

        self.generate_steps = False

    def forward(self, z):
        if self.generate_steps:
            return self.forward_with_steps(z)
        else:
            return self.forward_output_last(z)

    def forward_output_last(self, z):

        bsz = z.shape[0]  # (B, dim)

        # prepare initial hidden state. maybe use MLP?
        hc = self.latent_to_init(z)
        h, c = torch.split(
            hc, (self.num_layers * self.rnn_hidden_size,
                 self.num_layers * self.rnn_hidden_size), dim=-1)

        h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        h = h.transpose(0, 1)
        c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        c = c.transpose(0, 1)
        hc = (h.contiguous(), c.contiguous())

        # prepare inputs (len, B, dim)
        if self.use_pos_enc:
            # (len, 1, dim) + (1, B, dim)
            z = self.decoder_time_enc.unsqueeze(1) + z.unsqueeze(0)
        else:
            z = z.unsqueeze(0).expand(self.num_decoding_steps, -1, -1)

        h, _ = self.decoder_rnn(z, hc)
        z = self.kv_net(h)  # (len, B, dim)

        # outer products
        out = 0
        for t in range(self.num_decoding_steps):
            kx, vx = torch.split(
                z[t], (3 * self.im_size, 3 * self.im_size), dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)

            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out += x

        if self.out_tanh:
            return out.tanh()
        else:
            return out

    def forward_with_steps(self, z):

        bsz = z.shape[0]  # (B, dim)

        # prepare initial hidden state. maybe use MLP?
        hc = self.latent_to_init(z)
        h, c = torch.split(
            hc, (self.num_layers * self.rnn_hidden_size,
                 self.num_layers * self.rnn_hidden_size), dim=-1)

        h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        h = h.transpose(0, 1)
        c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        c = c.transpose(0, 1)
        hc = (h.contiguous(), c.contiguous())

        # prepare inputs (len, B, dim)
        if self.use_pos_enc:
            # (len, 1, dim) + (1, B, dim)
            z = self.decoder_time_enc.unsqueeze(1) + z.unsqueeze(0)
        else:
            z = z.unsqueeze(0).expand(self.num_decoding_steps, -1, -1)

        h, _ = self.decoder_rnn(z, hc)
        z = self.kv_net(h)  # (len, B, dim)

        step_list = []
        prefix_sum = []

        # compute norm
        out = 0
        for t in range(self.num_decoding_steps):
            kx, vx = torch.split(
                z[t], (3 * self.im_size, 3 * self.im_size), dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out += x
        normalizer = out.norm()

        # recompute
        out = 0
        for t in range(self.num_decoding_steps):
            kx, vx = torch.split(
                z[t], (3 * self.im_size, 3 * self.im_size), dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out += x
            step_list.append(x.clone() / x.norm() * normalizer)
            prefix_sum.append(out.clone() / out.norm() * normalizer)

        steps = torch.stack(step_list, dim=0).squeeze(1)
        prefix_sums = torch.stack(prefix_sum, dim=0).squeeze(1)
        steps = torch.cat([steps, prefix_sums], dim=0)
        return steps


# latent is split over time
class LSTMDeltaTimeSplitLatentGenerator(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 512,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers
        self.total_latent_dim = latent_dim

        self.rnn_init_fact = rnn_init_fact
        self.kvb_init_fact = kvb_init_fact

        assert latent_dim % num_decoding_steps == 0
        assert mini_latent_dim == latent_dim // num_decoding_steps
        self.mini_latent_dim = mini_latent_dim
        self.input_size = mini_input_size

        self.use_pos_enc = use_pos_enc  # hard coded
        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(
                    torch.randn(self.num_decoding_steps, mini_input_size)))

        self.per_time_linear_input = nn.Linear(mini_latent_dim, mini_input_size)

        self.decoder_rnn = nn.LSTM(mini_input_size, rnn_hidden_size, num_layers)

        # channels * 2 for key/value
        self.kvb_net = nn.Linear(
            rnn_hidden_size, init_channel * (2 * self.im_size + 1))

        self.custom_init_parameters()

    def custom_init_parameters(self):
        # times 10 for image related outputs
        bounds = 1.0 / math.sqrt(self.rnn_hidden_size) * self.rnn_init_fact
        for weight in self.decoder_rnn.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        dim = self.channels * (2 * self.im_size + 1)
        bounds = 1.0 / math.sqrt(dim) * self.kvb_init_fact
        for weight in self.kvb_net.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        nn.init.kaiming_normal_(
            self.per_time_linear_input.weight, a=0, mode='fan_in',
            nonlinearity='leaky_relu')

    def forward(self, z):

        bsz, total_dim = z.shape
        assert total_dim == self.total_latent_dim
        z = z.view(bsz, self.num_decoding_steps, self.mini_latent_dim)
        z = self.per_time_linear_input(z).transpose(0, 1)
        # (len, B, dim)
        if self.use_pos_enc:
            z = z + self.decoder_time_enc.unsqueeze(1)  # add pos

        z, _ = self.decoder_rnn(z)  # (len, b, dim)
        z = self.kvb_net(z)  # (len, B, ch * 2 * im_size)

        out = 0
        first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            lrs = lrs.reshape(bsz * self.channels, 1)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False

        if self.out_tanh:
            return out.tanh()
        else:
            return out


# use input generator
class LSTMOjaTimeSplitLatentGeneratorV2(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 512,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        v2_input_sigmoid = False,
        val_tanh = False,
        vect_lr = False,
        use_intermed_loss = 1,
        scale_intermed_loss = True,
        wide_out_dim = 0,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.v2_input_sigmoid = v2_input_sigmoid
        self.val_tanh = val_tanh
        self.vect_lr = vect_lr
        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers

        self.use_intermed_loss = use_intermed_loss
        self.scale_intermed_loss = scale_intermed_loss

        self.rnn_init_fact = rnn_init_fact
        self.kvb_init_fact = kvb_init_fact

        self.mini_latent_dim = mini_latent_dim
        self.input_size = mini_input_size

        self.use_pos_enc = use_pos_enc  # hard coded
        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(
                    torch.randn(self.num_decoding_steps, mini_input_size)))

        self.input_generator = nn.Linear(
            latent_dim, mini_latent_dim * num_decoding_steps)

        self.per_time_linear_input = nn.Linear(mini_latent_dim, mini_input_size)

        self.decoder_rnn = nn.LSTM(mini_input_size, rnn_hidden_size, num_layers)

        lr_dim = image_size if vect_lr else 1
        self.lr_dim = lr_dim

        self.wide_out_dim = wide_out_dim
        if wide_out_dim > 0:
            self.wide_net = nn.Sequential(
                nn.Linear(rnn_hidden_size, wide_out_dim),
                nn.ReLU(inplace=True))
            self.kvb_net = nn.Linear(
                wide_out_dim, init_channel * (2 * self.im_size + lr_dim))
        else:
            # channels * 2 for key/value
            self.kvb_net = nn.Linear(
                rnn_hidden_size, init_channel * (2 * self.im_size + lr_dim))

        self.custom_init_parameters()

    def custom_init_parameters(self):
        # times 10 for image related outputs
        bounds = 1.0 / math.sqrt(self.rnn_hidden_size) * self.rnn_init_fact
        for weight in self.decoder_rnn.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        dim = self.channels * (2 * self.im_size + 1)
        bounds = 1.0 / math.sqrt(dim) * self.kvb_init_fact
        for weight in self.kvb_net.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        nn.init.kaiming_normal_(
            self.per_time_linear_input.weight, a=0, mode='fan_in',
            nonlinearity='leaky_relu')

    def forward(self, z, output_only_last=True):

        bsz, _ = z.shape
        z = self.input_generator(z)
        if self.v2_input_sigmoid:
            z = torch.tanh(z)
        z = z.view(bsz, self.num_decoding_steps, self.mini_latent_dim)
        z = self.per_time_linear_input(z).transpose(0, 1)
        # (len, B, dim)
        if self.use_pos_enc:
            z = z + self.decoder_time_enc.unsqueeze(1)  # add pos

        z, _ = self.decoder_rnn(z)  # (len, b, dim)
        if self.wide_out_dim > 0:
            z = self.wide_net(z)
        z = self.kvb_net(z)  # (len, B, ch * 2 * im_size)

        out = 0
        first = True
        out_intermed = []
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply oja rule
            if not first:
                # Delta rule:
                kx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size,
                        self.im_size).transpose(1, 2),
                    vx.unsqueeze(2)).squeeze()
                kx = kx - kx_remove

            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)
            vx = vx * lrs

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            if self.use_intermed_loss > 1:
                for j in range(1, self.use_intermed_loss):
                    if t == j * self.num_decoding_steps // self.use_intermed_loss:
                        if self.scale_intermed_loss:
                            out_intermed.append(out.clone() * self.use_intermed_loss / j)
                        else:
                            out_intermed.append(out.clone())
            first = False

        if self.use_intermed_loss and not output_only_last:
            out_intermed.append(out)
            out = torch.cat(out_intermed, dim=0)

        if self.out_tanh:
            return out.tanh()
        else:
            return out


# use input generator
class LSTMDeltaTimeSplitLatentGeneratorV2(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 512,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        v2_input_sigmoid = False,
        val_tanh = False,
        vect_lr = False,
        use_latent_to_init = False,
        use_intermed_loss = 1,
        scale_intermed_loss = True,
        wide_out_dim = 0,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.v2_input_sigmoid = v2_input_sigmoid
        self.val_tanh = val_tanh
        self.vect_lr = vect_lr

        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers

        self.use_intermed_loss = use_intermed_loss
        self.scale_intermed_loss = scale_intermed_loss

        self.rnn_init_fact = rnn_init_fact
        self.kvb_init_fact = kvb_init_fact

        self.mini_latent_dim = mini_latent_dim
        self.input_size = mini_input_size

        self.use_pos_enc = use_pos_enc  # hard coded
        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(
                    torch.randn(self.num_decoding_steps, mini_input_size)))

        self.use_latent_to_init = use_latent_to_init
        if use_latent_to_init:
            self.latent_to_init = nn.Linear(
                latent_dim, 2 * rnn_hidden_size * num_layers)

        self.input_generator = nn.Linear(
            latent_dim, mini_latent_dim * num_decoding_steps)

        self.per_time_linear_input = nn.Linear(mini_latent_dim, mini_input_size)

        self.decoder_rnn = nn.LSTM(mini_input_size, rnn_hidden_size, num_layers)

        # learning rate dim
        lr_dim = image_size if vect_lr else 1
        self.lr_dim = lr_dim

        self.wide_out_dim = wide_out_dim
        if wide_out_dim > 0:
            self.wide_net = nn.Sequential(
                nn.Linear(rnn_hidden_size, wide_out_dim),
                nn.ReLU(inplace=True))
            self.kvb_net = nn.Linear(
                wide_out_dim, init_channel * (2 * self.im_size + lr_dim))
        else:
            # channels * 2 for key/value
            self.kvb_net = nn.Linear(
                rnn_hidden_size, init_channel * (2 * self.im_size + lr_dim))

        self.custom_init_parameters()

        self.generate_steps = False

    def custom_init_parameters(self):
        # times 10 for image related outputs
        bounds = 1.0 / math.sqrt(self.rnn_hidden_size) * self.rnn_init_fact
        for weight in self.decoder_rnn.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        dim = self.channels * (2 * self.im_size + 1)
        bounds = 1.0 / math.sqrt(dim) * self.kvb_init_fact
        for weight in self.kvb_net.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        nn.init.kaiming_normal_(
            self.per_time_linear_input.weight, a=0, mode='fan_in',
            nonlinearity='leaky_relu')

    def forward(self, z, output_only_last=True):
        if self.generate_steps:
            return self.forward_with_steps(z, output_only_last)
        else:
            return self.forward_output_last(z, output_only_last)

    def forward_output_last(self, z, output_only_last=True):

        bsz, _ = z.shape

        if self.use_latent_to_init:
            hc = self.latent_to_init(z)
            h, c = torch.split(
                hc, (self.num_layers * self.rnn_hidden_size,
                    self.num_layers * self.rnn_hidden_size), dim=-1)

            h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
            h = h.transpose(0, 1)
            c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
            c = c.transpose(0, 1)
            hc = (h.contiguous(), c.contiguous())


        z = self.input_generator(z)
        if self.v2_input_sigmoid:
            z = torch.tanh(z)
        z = z.view(bsz, self.num_decoding_steps, self.mini_latent_dim)
        z = self.per_time_linear_input(z).transpose(0, 1)
        # (len, B, dim)
        if self.use_pos_enc:
            z = z + self.decoder_time_enc.unsqueeze(1)  # add pos

        if self.use_latent_to_init:
            z, _ = self.decoder_rnn(z, hc)
        else:
            z, _ = self.decoder_rnn(z)  # (len, b, dim)
        if self.wide_out_dim > 0:
            z = self.wide_net(z)
        z = self.kvb_net(z)  # (len, B, ch * 2 * im_size)

        out = 0
        first = True
        out_intermed = []
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)

            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs *(vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x

            if self.use_intermed_loss > 1:
                for j in range(1, self.use_intermed_loss):
                    if t == j * self.num_decoding_steps // self.use_intermed_loss:
                        if self.scale_intermed_loss:
                            out_intermed.append(out.clone() * self.use_intermed_loss / j)
                        else:
                            out_intermed.append(out.clone())
            first = False

        if self.use_intermed_loss and not output_only_last:
            out_intermed.append(out)
            out = torch.cat(out_intermed, dim=0)

        if self.out_tanh:
            return out.tanh()
        else:
            return out

    def forward_with_steps(self, z, output_only_last=True):

        bsz, _ = z.shape

        if self.use_latent_to_init:
            hc = self.latent_to_init(z)
            h, c = torch.split(
                hc, (self.num_layers * self.rnn_hidden_size,
                    self.num_layers * self.rnn_hidden_size), dim=-1)

            h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
            h = h.transpose(0, 1)
            c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
            c = c.transpose(0, 1)
            hc = (h.contiguous(), c.contiguous())


        z = self.input_generator(z)
        if self.v2_input_sigmoid:
            z = torch.tanh(z)
        z = z.view(bsz, self.num_decoding_steps, self.mini_latent_dim)
        z = self.per_time_linear_input(z).transpose(0, 1)
        # (len, B, dim)
        if self.use_pos_enc:
            z = z + self.decoder_time_enc.unsqueeze(1)  # add pos

        if self.use_latent_to_init:
            z, _ = self.decoder_rnn(z, hc)
        else:
            z, _ = self.decoder_rnn(z)  # (len, b, dim)
        if self.wide_out_dim > 0:
            z = self.wide_net(z)
        z = self.kvb_net(z)  # (len, B, ch * 2 * im_size)

        step_list = []
        prefix_sum = []

        # compute normalizer
        out = 0
        first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)

            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs *(vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False

        normalizer = out.norm()

        # recompute
        out = 0
        first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)

            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs *(vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            if self.out_tanh:
                step_list.append((x.clone() / x.norm() * normalizer).tanh())
                prefix_sum.append((out.clone() / out.norm() * normalizer).tanh())
            else:
                step_list.append(x.clone() / x.norm() * normalizer)
                prefix_sum.append(out.clone() / out.norm() * normalizer)
            first = False

        steps = torch.stack(step_list, dim=0).squeeze(1)
        prefix_sums = torch.stack(prefix_sum, dim=0).squeeze(1)
        steps = torch.cat([steps, prefix_sums], dim=0)
        return steps


class LSTMDeltaGenerator(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 256,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        val_tanh = False,
        vect_lr = False,
        use_learned_init = False,
        use_intermed_loss = 1,
        scale_intermed_loss = True,
        wide_out_dim = 0,
        clip = 0.0,
        clip_rnn = 0.0,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.val_tanh = val_tanh
        self.vect_lr = vect_lr

        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers
        self.wide_out_dim = wide_out_dim
        self.use_intermed_loss = use_intermed_loss
        self.scale_intermed_loss = scale_intermed_loss

        self.clip = clip
        self.clip_rnn = clip_rnn

        self.rnn_init_fact = rnn_init_fact
        self.kvb_init_fact = kvb_init_fact

        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(
                    torch.randn(self.num_decoding_steps, latent_dim)))

        self.use_learned_init = use_learned_init
        if use_learned_init:
            self.register_parameter(
                name='learned_init',
                param=torch.nn.Parameter(
                    torch.randn(self.channels, self.im_size, self.im_size)))

        # map latent to LSTM initial states
        self.latent_to_init = nn.Linear(
            latent_dim, 2 * rnn_hidden_size * num_layers)

        self.decoder_rnn = nn.LSTM(latent_dim, rnn_hidden_size, num_layers)
        # channels * 2 for key/value

        # learning rate dim
        lr_dim = image_size if vect_lr else 1
        self.lr_dim = lr_dim

        self.wide_out_dim = wide_out_dim
        if wide_out_dim > 0:
            self.wide_net = nn.Sequential(
                nn.Linear(rnn_hidden_size, wide_out_dim),
                nn.ReLU(inplace=True))
            self.kvb_net = nn.Linear(
                wide_out_dim, init_channel * (2 * self.im_size + lr_dim))
        else:
            self.kvb_net = nn.Linear(
                rnn_hidden_size, init_channel * (2 * self.im_size + lr_dim))

        self.generate_steps = False

        self.custom_init_parameters()

    def custom_init_parameters(self):
        # times 10 for image related outputs
        bounds = 1.0 / math.sqrt(self.rnn_hidden_size) * self.rnn_init_fact
        for weight in self.decoder_rnn.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        dim = self.channels * (2 * self.im_size + 1)
        bounds = 1.0 / math.sqrt(dim) * self.kvb_init_fact
        for weight in self.kvb_net.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        nn.init.kaiming_normal_(
            self.latent_to_init.weight, a=0, mode='fan_in',
            nonlinearity='leaky_relu')

    def forward(self, z, output_only_last=True):
        if self.generate_steps:
            return self.forward_with_steps(z)
        else:
            return self.forward_output_last(z, output_only_last)

    def forward_output_last(self, z, output_only_last=True):
        bsz = z.shape[0]  # (B, dim)

        # prepare initial hidden state. maybe use MLP?
        hc = self.latent_to_init(z)
        h, c = torch.split(
            hc, (self.num_layers * self.rnn_hidden_size,
                 self.num_layers * self.rnn_hidden_size), dim=-1)

        h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        h = h.transpose(0, 1)
        c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        c = c.transpose(0, 1)
        hc = (h.contiguous(), c.contiguous())

        # prepare inputs (len, B, dim)
        if self.use_pos_enc:
            # (len, 1, dim) + (1, B, dim)
            z = self.decoder_time_enc.unsqueeze(1) + z.unsqueeze(0)
        else:
            z = z.unsqueeze(0).expand(self.num_decoding_steps, -1, -1)

        h, _ = self.decoder_rnn(z, hc)
        if self.wide_out_dim > 0:
            h = self.wide_net(h)
        z = self.kvb_net(h)  # (len, B, dim)

        # outer products
        if self.use_learned_init:
            out = self.learned_init.unsqueeze(0).repeat(bsz, 1, 1, 1)
            first = False
        else:
            out = 0
            first = True
        out_intermed = []
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            if self.use_intermed_loss > 1:
                for j in range(1, self.use_intermed_loss):
                    if t == j * self.num_decoding_steps // self.use_intermed_loss:
                        # normalize by number of steps
                        if self.scale_intermed_loss:
                            out_intermed.append(out.clone() * self.use_intermed_loss / j)
                        else:
                            out_intermed.append(out.clone())
            first = False

        if self.use_intermed_loss and not output_only_last:
            out_intermed.append(out)
            out = torch.cat(out_intermed, dim=0)

        if self.out_tanh:
            return out.tanh()
        else:
            return out

    def forward_with_steps(self, z):
        bsz = z.shape[0]  # (B, dim)

        # prepare initial hidden state. maybe use MLP?
        hc = self.latent_to_init(z)
        h, c = torch.split(
            hc, (self.num_layers * self.rnn_hidden_size,
                 self.num_layers * self.rnn_hidden_size), dim=-1)

        h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        h = h.transpose(0, 1)
        c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        c = c.transpose(0, 1)
        hc = (h.contiguous(), c.contiguous())

        # prepare inputs (len, B, dim)
        if self.use_pos_enc:
            # (len, 1, dim) + (1, B, dim)
            z = self.decoder_time_enc.unsqueeze(1) + z.unsqueeze(0)
        else:
            z = z.unsqueeze(0).expand(self.num_decoding_steps, -1, -1)

        h, _ = self.decoder_rnn(z, hc)
        if self.wide_out_dim > 0:
            h = self.wide_net(h)
        z = self.kvb_net(h)  # (len, B, dim)

        # outer products
        if self.use_learned_init:
            out = self.learned_init.unsqueeze(0).repeat(bsz, 1, 1, 1)
            first = False
        else:
            out = 0
            first = True
        step_raw = []
        step_list_nolr = []
        step_list = []
        prefix_sum = []
        # compute norm
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)
            x_ = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = (vx - vx_remove)

            x_no_lr = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            if not first:
                vx = lrs * vx
            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False
        normalizer = out.norm()

        out = 0
        first = True
        # recompute
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)
            x_ = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            step_raw.append(x_.clone() / x_.norm() * normalizer)
            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = (vx - vx_remove)

            x_no_lr = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            step_list_nolr.append(x_no_lr / x_no_lr.norm() * normalizer)
            if not first:
                vx = lrs * vx
            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            step_list.append(x.clone() / x.norm() * normalizer)
            prefix_sum.append(out.clone() / out.norm() * normalizer)
            first = False

        steps = torch.stack(step_list, dim=0).squeeze(1)
        prefix_sums = torch.stack(prefix_sum, dim=0).squeeze(1)
        steps = torch.cat([steps, prefix_sums], dim=0)
        return steps


class RFWPDeltaGenerator(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 256,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        use_learned_init = False,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers

        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(

                    torch.randn(self.num_decoding_steps, latent_dim)))
        self.use_learned_init = use_learned_init
        if use_learned_init:
            self.register_parameter(
                name='learned_init',
                param=torch.nn.Parameter(
                    torch.randn(self.channels, self.im_size, self.im_size)))

        # map latent to RNN initial states
        self.latent_to_init = nn.Linear(
            latent_dim, rnn_hidden_size * num_layers)

        self.latent_to_input = nn.Linear(latent_dim, rnn_hidden_size)

        # RFWP
        self.d_model = rnn_hidden_size
        assert self.d_model % num_heads == 0
        head_dim = self.d_model // num_heads

        self.num_head = num_heads
        self.dim_head = head_dim

        rfwp_layers = []
        ff_layers = []
        for _ in range(num_layers):
            rfwp_layers.append(
                RFWPlayer(num_heads, head_dim, self.d_model, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_feedforward, self.d_model, dropout))
        self.rfwp_layers = nn.ModuleList(rfwp_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

        # channels * 2 for key/value
        self.kvb_net = nn.Linear(
            rnn_hidden_size, init_channel * (2 * self.im_size + 1))

    def forward(self, z):
        bsz = z.shape[0]  # (B, dim)

        h = self.latent_to_init(z)
        h = h.reshape(bsz, self.num_layers, self.num_head, 1, self.dim_head)
        h = h.transpose(0, 1)  # (layers, B, D)

        # prepare inputs (len, B, dim)
        if self.use_pos_enc:
            # (len, 1, dim) + (1, B, dim)
            z = self.decoder_time_enc.unsqueeze(1) + z.unsqueeze(0)
        else:
            z = z.unsqueeze(0).expand(self.num_decoding_steps, -1, -1)

        z = self.latent_to_input(z)
        lay_id = 0
        for rfwp_layer, ff_layer in zip(self.rfwp_layers, self.ff_layers):
            z = rfwp_layer(z, h[lay_id])
            z = ff_layer(z)
            lay_id += 1
        z = self.kvb_net(z)  # (len, B, dim)

        # outer products
        if self.use_learned_init:
            out = self.learned_init.unsqueeze(0).repeat(bsz, 1, 1, 1)
            first = False
        else:
            out = 0
            first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            lrs = lrs.reshape(bsz * self.channels, 1)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False

        if self.out_tanh:
            return out.tanh()
        else:
            return out

# Unet augmented models
class LSTMDeltaGeneratorUnet(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 256,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        stop_grad_unet = False,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        use_learned_init = False,
        clip = 0.0,
        clip_rnn = 0.0,
        max_unet_dim = 4,
        frozen_base = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
        unet_tanh = False,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers

        self.unet_tanh = unet_tanh

        self.clip = clip
        self.clip_rnn = clip_rnn

        self.rnn_init_fact = rnn_init_fact
        self.kvb_init_fact = kvb_init_fact

        self.stop_grad_unet = stop_grad_unet
        self.frozen_base = frozen_base
        self.use_pos_enc = use_pos_enc

        base_module_list = []
        unet_module_list = []

        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(
                    torch.randn(self.num_decoding_steps, latent_dim)))
            base_module_list.append(self.decoder_time_enc)

        self.use_learned_init = use_learned_init
        if use_learned_init:
            self.register_parameter(
                name='learned_init',
                param=torch.nn.Parameter(
                    torch.randn(self.channels, self.im_size, self.im_size)))
            base_module_list.append(self.learned_init)

        # map latent to LSTM initial states
        self.latent_to_init = nn.Linear(
            latent_dim, 2 * rnn_hidden_size * num_layers)
        base_module_list.append(self.latent_to_init)

        self.decoder_rnn = nn.LSTM(latent_dim, rnn_hidden_size, num_layers)
        base_module_list.append(self.decoder_rnn)
        # channels * 2 for key/value
        self.kvb_net = nn.Linear(
            rnn_hidden_size, init_channel * (2 * self.im_size + 1))
        base_module_list.append(self.kvb_net)

        unet_dims = []
        k = 1
        while k <= max_unet_dim:
            unet_dims.append(k)
            k = 2 * k
        unet_dims = tuple(unet_dims)

        self.u_net = Unet(dim=image_size,
                          channels=init_channel,
                          dim_mults=unet_dims)
        unet_module_list.append(self.u_net)

        # https://stackoverflow.com/questions/69774137/constructing-parameter-groups-in-pytorch
        self.mod_dict = torch.nn.ModuleDict({
            'base': torch.nn.ModuleList(base_module_list),
            'unet': torch.nn.ModuleList(unet_module_list)})

        self.output = 'not_specified'

        if frozen_base:
            for p in self.mod_dict['base'].parameters():
                p.requires_grad = False

        self.custom_init_parameters()

        self.generate_steps = False

    def custom_init_parameters(self):
        # times 10 for image related outputs
        bounds = 1.0 / math.sqrt(self.rnn_hidden_size) * self.rnn_init_fact
        for weight in self.decoder_rnn.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        dim = self.channels * (2 * self.im_size + 1)
        bounds = 1.0 / math.sqrt(dim) * self.kvb_init_fact
        for weight in self.kvb_net.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        nn.init.kaiming_normal_(
            self.latent_to_init.weight, a=0, mode='fan_in',
            nonlinearity='leaky_relu')

    def forward(self, z):
        if self.generate_steps:
            return self.forward_with_steps(z)
        else:
            return self.forward_output_last(z)

    def forward_output_last(self, z, use_pre_unet_loss=False):
        bsz = z.shape[0]  # (B, dim)

        hc = self.latent_to_init(z)
        h, c = torch.split(
            hc, (self.num_layers * self.rnn_hidden_size,
                 self.num_layers * self.rnn_hidden_size), dim=-1)

        h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        h = h.transpose(0, 1)
        c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        c = c.transpose(0, 1)
        hc = (h.contiguous(), c.contiguous())

        # prepare inputs (len, B, dim)
        if self.use_pos_enc:
            # (len, 1, dim) + (1, B, dim)
            z = self.decoder_time_enc.unsqueeze(1) + z.unsqueeze(0)
        else:
            z = z.unsqueeze(0).expand(self.num_decoding_steps, -1, -1)

        h, _ = self.decoder_rnn(z, hc)
        z = self.kvb_net(h)  # (len, B, dim)

        # outer products
        if self.use_learned_init:
            out = self.learned_init.unsqueeze(0).repeat(bsz, 1, 1, 1)
            first = False
        else:
            out = 0
            first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            lrs = lrs.reshape(bsz * self.channels, 1)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False

        if self.output == 'pre_unet':
            if self.out_tanh:
                return out.tanh()
            else:
                return out

        if self.output == 'post_unet':
            if self.unet_tanh:
                out = out.tanh()
            out = self.u_net(out)
            if self.out_tanh:
                return out.tanh()
            else:
                return out

        if use_pre_unet_loss:
            if self.stop_grad_unet:
                if self.unet_tanh:
                    out_unet = self.u_net(out.tanh().detach())
                else:
                    out_unet = self.u_net(out.detach())
            else:
                if self.unet_tanh:
                    out_unet = self.u_net(out.tanh())
                else:
                    out_unet = self.u_net(out)
            out = torch.cat([out, out_unet], dim=0)
        else:
            # apply unet refinement
            if self.unet_tanh:
                out = out.tanh()
            if self.stop_grad_unet:
                out = self.u_net(out.detach())
            else:
                out = self.u_net(out)

        if self.out_tanh:
            return out.tanh()
        else:
            return out


    def forward_with_steps(self, z, use_pre_unet_loss=False):
        bsz = z.shape[0]  # (B, dim)

        hc = self.latent_to_init(z)
        h, c = torch.split(
            hc, (self.num_layers * self.rnn_hidden_size,
                 self.num_layers * self.rnn_hidden_size), dim=-1)

        h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        h = h.transpose(0, 1)
        c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
        c = c.transpose(0, 1)
        hc = (h.contiguous(), c.contiguous())

        # prepare inputs (len, B, dim)
        if self.use_pos_enc:
            # (len, 1, dim) + (1, B, dim)
            z = self.decoder_time_enc.unsqueeze(1) + z.unsqueeze(0)
        else:
            z = z.unsqueeze(0).expand(self.num_decoding_steps, -1, -1)

        h, _ = self.decoder_rnn(z, hc)
        z = self.kvb_net(h)  # (len, B, dim)

        step_list = []
        prefix_sum = []

        # compute norm
        if self.use_learned_init:
            out = self.learned_init.unsqueeze(0).repeat(bsz, 1, 1, 1)
            first = False
        else:
            out = 0
            first = True

        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            lrs = lrs.reshape(bsz * self.channels, 1)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False
        normalizer = out.norm()

        # recompute
        if self.use_learned_init:
            out = self.learned_init.unsqueeze(0).repeat(bsz, 1, 1, 1)
            first = False
        else:
            out = 0
            first = True

        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            lrs = lrs.reshape(bsz * self.channels, 1)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            if self.out_tanh:
                step_list.append((x.clone() / x.norm() * normalizer).tanh())
                prefix_sum.append((out.clone() / out.norm() * normalizer).tanh())
            else:
                step_list.append(x.clone() / x.norm() * normalizer)
                prefix_sum.append(out.clone() / out.norm() * normalizer)
            first = False

        unet_out = self.u_net(out)
        if self.out_tanh:
            step_list.append((out - unet_out).tanh())
            prefix_sum.append(unet_out.tanh())
        else:
            step_list.append((out - unet_out).tanh())
            prefix_sum.append(unet_out.tanh())

        steps = torch.stack(step_list, dim=0).squeeze(1)
        prefix_sums = torch.stack(prefix_sum, dim=0).squeeze(1)
        # steps = torch.cat([raw_steps, nolr_steps, steps, prefix_sums], dim=0)
        steps = torch.cat([steps, prefix_sums], dim=0)
        return steps


class LSTMDeltaTimeSplitLatentGeneratorUnet(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 512,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        stop_grad_unet = False,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        max_unet_dim = 4,
        frozen_base = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers
        self.total_latent_dim = latent_dim
        self.frozen_base = frozen_base

        self.rnn_init_fact = rnn_init_fact
        self.kvb_init_fact = kvb_init_fact

        assert latent_dim % num_decoding_steps == 0
        assert mini_latent_dim == latent_dim // num_decoding_steps
        self.mini_latent_dim = mini_latent_dim
        self.input_size = mini_input_size

        self.stop_grad_unet = stop_grad_unet
        self.use_pos_enc = use_pos_enc  # hard coded

        base_module_list = []
        unet_module_list = []

        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(
                    torch.randn(self.num_decoding_steps, mini_input_size)))
            base_module_list.append(self.decoder_time_enc)

        self.per_time_linear_input = nn.Linear(mini_latent_dim, mini_input_size)
        base_module_list.append(self.per_time_linear_input)

        self.decoder_rnn = nn.LSTM(mini_input_size, rnn_hidden_size, num_layers)
        base_module_list.append(self.decoder_rnn)

        # channels * 2 for key/value
        self.kvb_net = nn.Linear(
            rnn_hidden_size, init_channel * (2 * self.im_size + 1))
        base_module_list.append(self.kvb_net)

        unet_dims = []
        k = 1
        while k <= max_unet_dim:
            unet_dims.append(k)
            k = 2 * k
        unet_dims = tuple(unet_dims)

        self.u_net = Unet(dim=image_size,
                          channels=init_channel,
                          dim_mults=unet_dims)
        unet_module_list.append(self.u_net)

        # https://stackoverflow.com/questions/69774137/constructing-parameter-groups-in-pytorch
        self.mod_dict = torch.nn.ModuleDict({
            'base': torch.nn.ModuleList(base_module_list),
            'unet': torch.nn.ModuleList(unet_module_list)})

        self.output = 'not_specified'

        if frozen_base:
            for p in self.mod_dict['base'].parameters():
                p.requires_grad = False

        self.custom_init_parameters()

    def custom_init_parameters(self):
        # times 10 for image related outputs
        bounds = 1.0 / math.sqrt(self.rnn_hidden_size) * self.rnn_init_fact
        for weight in self.decoder_rnn.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        dim = self.channels * (2 * self.im_size + 1)
        bounds = 1.0 / math.sqrt(dim) * self.kvb_init_fact
        for weight in self.kvb_net.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        nn.init.kaiming_normal_(
            self.per_time_linear_input.weight, a=0, mode='fan_in',
            nonlinearity='leaky_relu')

    def forward(self, z, use_pre_unet_loss=False):

        bsz, total_dim = z.shape
        assert total_dim == self.total_latent_dim
        z = z.view(bsz, self.num_decoding_steps, self.mini_latent_dim)
        z = self.per_time_linear_input(z).transpose(0, 1)
        # (len, B, dim)
        if self.use_pos_enc:
            z = z + self.decoder_time_enc.unsqueeze(1)  # add pos

        z, _ = self.decoder_rnn(z)  # (len, b, dim)
        z = self.kvb_net(z)  # (len, B, ch * 2 * im_size)

        out = 0
        first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            lrs = lrs.reshape(bsz * self.channels, 1)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False

        if self.output == 'pre_unet':
            if self.out_tanh:
                return out.tanh()
            else:
                return out

        if self.output == 'post_unet':
            out = self.u_net(out)
            if self.out_tanh:
                return out.tanh()
            else:
                return out

        if use_pre_unet_loss:
            if self.stop_grad_unet:
                out_unet = self.u_net(out.detach())
            else:
                out_unet = self.u_net(out)
            out = torch.cat([out, out_unet], dim=0)
        else:
            # apply unet refinement
            if self.stop_grad_unet:
                out = self.u_net(out.detach())
            else:
                out = self.u_net(out)

        if self.out_tanh:
            return out.tanh()
        else:
            return out


class LSTMDeltaTimeSplitLatentGeneratorV2Unet(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 512,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        stop_grad_unet = False,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        v2_input_sigmoid = False,
        val_tanh = False,
        vect_lr = False,
        use_latent_to_init = False,
        use_intermed_loss = 1,
        scale_intermed_loss = True,
        wide_out_dim = 0,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        max_unet_dim = 4,
        frozen_base = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
        unet_tanh = False,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.v2_input_sigmoid = v2_input_sigmoid
        self.val_tanh = val_tanh
        self.vect_lr = vect_lr

        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers

        self.total_latent_dim = latent_dim
        self.frozen_base = frozen_base

        self.use_intermed_loss = use_intermed_loss
        self.scale_intermed_loss = scale_intermed_loss

        self.rnn_init_fact = rnn_init_fact
        self.kvb_init_fact = kvb_init_fact

        self.unet_tanh = unet_tanh

        self.mini_latent_dim = mini_latent_dim
        self.input_size = mini_input_size

        self.stop_grad_unet = stop_grad_unet
        self.use_pos_enc = use_pos_enc  # hard coded

        base_module_list = []
        unet_module_list = []

        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(
                    torch.randn(self.num_decoding_steps, mini_input_size)))
            base_module_list.append(self.decoder_time_enc)

        self.use_latent_to_init = use_latent_to_init
        if use_latent_to_init:
            self.latent_to_init = nn.Linear(
                latent_dim, 2 * rnn_hidden_size * num_layers)
            base_module_list.append(self.latent_to_init)

        self.input_generator = nn.Linear(
            latent_dim, mini_latent_dim * num_decoding_steps)
        base_module_list.append(self.input_generator)

        self.per_time_linear_input = nn.Linear(mini_latent_dim, mini_input_size)
        base_module_list.append(self.per_time_linear_input)

        self.decoder_rnn = nn.LSTM(mini_input_size, rnn_hidden_size, num_layers)
        base_module_list.append(self.decoder_rnn)

        # learning rate dim
        lr_dim = image_size if vect_lr else 1
        self.lr_dim = lr_dim

        # channels * 2 for key/value
        self.kvb_net = nn.Linear(
            rnn_hidden_size, init_channel * (2 * self.im_size + lr_dim))
        base_module_list.append(self.kvb_net)

        unet_dims = []
        k = 1
        while k <= max_unet_dim:
            unet_dims.append(k)
            k = 2 * k
        unet_dims = tuple(unet_dims)

        self.u_net = Unet(dim=image_size,
                          channels=init_channel,
                          dim_mults=unet_dims)
        unet_module_list.append(self.u_net)

        # https://stackoverflow.com/questions/69774137/constructing-parameter-groups-in-pytorch
        self.mod_dict = torch.nn.ModuleDict({
            'base': torch.nn.ModuleList(base_module_list),
            'unet': torch.nn.ModuleList(unet_module_list)})

        self.output = 'not_specified'

        if frozen_base:
            for p in self.mod_dict['base'].parameters():
                p.requires_grad = False

        self.generate_steps = False

        self.custom_init_parameters()

    def custom_init_parameters(self):
        # times 10 for image related outputs
        bounds = 1.0 / math.sqrt(self.rnn_hidden_size) * self.rnn_init_fact
        for weight in self.decoder_rnn.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        dim = self.channels * (2 * self.im_size + 1)
        bounds = 1.0 / math.sqrt(dim) * self.kvb_init_fact
        for weight in self.kvb_net.parameters():
            nn.init.uniform_(weight, -bounds, bounds)

        nn.init.kaiming_normal_(
            self.per_time_linear_input.weight, a=0, mode='fan_in',
            nonlinearity='leaky_relu')

    def forward(self, z, use_pre_unet_loss=False):
        if self.generate_steps:
            return self.forward_with_steps(z, use_pre_unet_loss)
        else:
            return self.forward_output_last(z, use_pre_unet_loss)


    def forward_output_last(self, z, use_pre_unet_loss=False):

        bsz, _ = z.shape

        if self.use_latent_to_init:
            hc = self.latent_to_init(z)
            h, c = torch.split(
                hc, (self.num_layers * self.rnn_hidden_size,
                    self.num_layers * self.rnn_hidden_size), dim=-1)

            h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
            h = h.transpose(0, 1)
            c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
            c = c.transpose(0, 1)
            hc = (h.contiguous(), c.contiguous())

        z = self.input_generator(z)
        if self.v2_input_sigmoid:
            z = torch.tanh(z)
        z = z.view(bsz, self.num_decoding_steps, self.mini_latent_dim)
        z = self.per_time_linear_input(z).transpose(0, 1)
        # (len, B, dim)
        if self.use_pos_enc:
            z = z + self.decoder_time_enc.unsqueeze(1)  # add pos

        if self.use_latent_to_init:
            z, _ = self.decoder_rnn(z, hc)  # (len, b, dim)
        else:
            z, _ = self.decoder_rnn(z)  # (len, b, dim)
        z = self.kvb_net(z)  # (len, B, ch * 2 * im_size)

        out = 0
        first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False

        if self.output == 'pre_unet':
            if self.out_tanh:
                return out.tanh()
            else:
                return out

        if self.output == 'post_unet':
            if self.unet_tanh:
                out = out.tanh()
            out = self.u_net(out)
            if self.out_tanh:
                return out.tanh()
            else:
                return out

        if self.unet_tanh:
            out_unet = out.tanh()
        else:
            out_unet = out

        if use_pre_unet_loss:
            if self.stop_grad_unet:
                out_unet = self.u_net(out_unet.detach())
            else:
                out_unet = self.u_net(out_unet)
            out = torch.cat([out, out_unet], dim=0)
        else:
            # apply unet refinement
            if self.stop_grad_unet:
                out = self.u_net(out_unet.detach())
            else:
                out = self.u_net(out_unet)

        if self.out_tanh:
            return out.tanh()
        else:
            return out


    def forward_with_steps(self, z, use_pre_unet_loss=False):

        bsz, _ = z.shape

        if self.use_latent_to_init:
            hc = self.latent_to_init(z)
            h, c = torch.split(
                hc, (self.num_layers * self.rnn_hidden_size,
                    self.num_layers * self.rnn_hidden_size), dim=-1)

            h = h.reshape(bsz, self.num_layers, self.rnn_hidden_size)
            h = h.transpose(0, 1)
            c = c.reshape(bsz, self.num_layers, self.rnn_hidden_size)
            c = c.transpose(0, 1)
            hc = (h.contiguous(), c.contiguous())

        z = self.input_generator(z)
        if self.v2_input_sigmoid:
            z = torch.tanh(z)
        z = z.view(bsz, self.num_decoding_steps, self.mini_latent_dim)
        z = self.per_time_linear_input(z).transpose(0, 1)
        # (len, B, dim)
        if self.use_pos_enc:
            z = z + self.decoder_time_enc.unsqueeze(1)  # add pos

        if self.use_latent_to_init:
            z, _ = self.decoder_rnn(z, hc)  # (len, b, dim)
        else:
            z, _ = self.decoder_rnn(z)  # (len, b, dim)
        z = self.kvb_net(z)  # (len, B, ch * 2 * im_size)

        step_list = []
        prefix_sum = []

        # compute normalizer
        out = 0
        first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False

        normalizer = out.norm()

        # recompute
        out = 0
        first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels * self.lr_dim),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            if self.val_tanh:
                vx = torch.tanh(vx)
            lrs = lrs.reshape(bsz * self.channels, self.lr_dim)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            if self.out_tanh:
                step_list.append((x.clone() / x.norm() * normalizer).tanh())
                prefix_sum.append((out.clone() / out.norm() * normalizer).tanh())
            else:
                step_list.append(x.clone() / x.norm() * normalizer)
                prefix_sum.append(out.clone() / out.norm() * normalizer)

            first = False

        unet_out = self.u_net(out)
        if self.out_tanh:
            step_list.append((out - unet_out).tanh())
            prefix_sum.append(unet_out.tanh())
        else:
            step_list.append((out - unet_out).tanh())
            prefix_sum.append(unet_out.tanh())

        steps = torch.stack(step_list, dim=0).squeeze(1)
        prefix_sums = torch.stack(prefix_sum, dim=0).squeeze(1)
        # steps = torch.cat([raw_steps, nolr_steps, steps, prefix_sums], dim=0)
        steps = torch.cat([steps, prefix_sums], dim=0)
        return steps


class TransformerDeltaTimeSplitLatentGenerator(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 512,
        mini_latent_dim = 8,  # per time latent dim
        rnn_hidden_size = 512,
        num_layers = 1,
        num_decoding_steps = 64,
        transparent = False,
        greyscale = False,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
    ):
        super().__init__()
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.channels = init_channel
        self.num_decoding_steps = num_decoding_steps
        self.rnn_hidden_size = rnn_hidden_size
        self.im_size = image_size
        self.out_tanh = out_tanh
        self.num_layers = num_layers
        self.total_latent_dim = latent_dim

        assert latent_dim % num_decoding_steps == 0
        assert mini_latent_dim == latent_dim // num_decoding_steps
        self.mini_latent_dim = mini_latent_dim
        self.d_model = rnn_hidden_size

        self.per_time_linear_input = nn.Linear(mini_latent_dim, rnn_hidden_size)

        self.use_pos_enc = use_pos_enc  # hard coded
        if use_pos_enc:
            self.register_parameter(
                name='decoder_time_enc',
                param=torch.nn.Parameter(
                    torch.randn(self.num_decoding_steps, rnn_hidden_size)))

        # Transformer
        assert self.d_model % num_heads == 0
        head_dim = self.d_model // num_heads
        layers = []
        for _ in range(num_layers):
            layers.append(
                Attention(num_heads, self.d_model, head_dim, dropout))
            layers.append(
                TransformerFFlayers(dim_feedforward, self.d_model, dropout))
        self.transformer_encoder = nn.Sequential(*layers)

        # channels * 2 for key/value
        self.kvb_net = nn.Linear(
            self.d_model, init_channel * (2 * self.im_size + 1))

    def forward(self, z):
        bsz, total_dim = z.shape
        assert total_dim == self.total_latent_dim

        z = z.view(bsz, self.num_decoding_steps, self.mini_latent_dim)
        z = self.per_time_linear_input(z).transpose(0, 1)
        # (len, B, dim)
        if self.use_pos_enc:
            z = z + self.decoder_time_enc.unsqueeze(1)  # add pos

        z = self.transformer_encoder(z)  # (len, b, dim)
        z = self.kvb_net(z)  # (len, B, ch * 2 * im_size)

        out = 0
        first = True
        for t in range(self.num_decoding_steps):
            kx, vx, lrs = torch.split(
                z[t],
                (self.channels * self.im_size,
                 self.channels * self.im_size,
                 self.channels),
                dim=-1)
            kx = kx.reshape(bsz * self.channels, self.im_size)
            vx = vx.reshape(bsz * self.channels, self.im_size)
            lrs = lrs.reshape(bsz * self.channels, 1)
            if not self.no_sigmoid_lr:
                lrs = torch.sigmoid(lrs)
            if self.use_softmax:
                kx = torch.softmax(kx, dim=-1)

            # apply delta rule
            if not first:
                # Delta rule:
                vx_remove = torch.bmm(
                    out.reshape(
                        bsz * self.channels, self.im_size, self.im_size),
                    kx.unsqueeze(2)).squeeze()
                vx = lrs * (vx - vx_remove)

            x = torch.bmm(vx.unsqueeze(2), kx.unsqueeze(1))
            x = x.view(bsz, self.channels, self.im_size, self.im_size)
            out = out + x
            first = False

        if self.out_tanh:
            return out.tanh()
        else:
            return out


###############################################################################


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out = 3,
        num_upsamples = 4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in

        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if not last_layer else final_chan * 2
            layer = nn.Sequential(
                upsample(),
                nn.Conv2d(chans, chan_out, 3, padding = 1),
                nn.GLU(dim = 1)
            )
            self.layers.append(layer)
            chans //= 2

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(BaseModel):
    def __init__(
        self,
        *,
        image_size,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = []
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        assert disc_output_size in {1, 5}, 'discriminator output dimensions can only be 5x5 or 1x1'

        resolution = int(resolution)

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        non_residual_resolutions = range(min(8, resolution), 2, -1)
        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), non_residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))

        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        chan_in_out = list(zip(features[:-1], features[1:]))

        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(nn.Sequential(
                Blur(),
                nn.Conv2d(init_channel, chan_out, 4, stride = 2, padding = 1),
                nn.LeakyReLU(0.1)
            ))

        self.residual_layers = nn.ModuleList([])

        for (res, ((_, chan_in), (_, chan_out))) in zip(non_residual_resolutions, chan_in_out):
            image_width = 2 ** res

            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            self.residual_layers.append(nn.ModuleList([
                SumBranches([
                    nn.Sequential(
                        Blur(),
                        nn.Conv2d(chan_in, chan_out, 4, stride = 2, padding = 1),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(chan_out, chan_out, 3, padding = 1),
                        nn.LeakyReLU(0.1)
                    ),
                    nn.Sequential(
                        Blur(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(chan_in, chan_out, 1),
                        nn.LeakyReLU(0.1),
                    )
                ]),
                attn
            ]))

        last_chan = features[-1][-1]
        if disc_output_size == 5:
            self.to_logits = nn.Sequential(
                nn.Conv2d(last_chan, last_chan, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )
        elif disc_output_size == 1:
            self.to_logits = nn.Sequential(
                Blur(),
                nn.Conv2d(last_chan, last_chan, 3, stride = 2, padding = 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )

        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(init_channel, 64, 3, padding = 1),
            Residual(PreNorm(64, LinearAttention(64))),
            SumBranches([
                nn.Sequential(
                    Blur(),
                    nn.Conv2d(64, 32, 4, stride = 2, padding = 1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(32, 32, 3, padding = 1),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    Blur(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(64, 32, 1),
                    nn.LeakyReLU(0.1),
                )
            ]),
            Residual(PreNorm(32, LinearAttention(32))),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(32, 1, 4)
        )

        self.decoder1 = SimpleDecoder(chan_in = last_chan, chan_out = init_channel)
        self.decoder2 = SimpleDecoder(chan_in = features[-2][-1], chan_out = init_channel) if resolution >= 9 else None

    def forward(self, x, calc_aux_loss = False):
        orig_img = x

        for layer in self.non_residual_layers:
            x = layer(x)

        layer_outputs = []

        for (net, attn) in self.residual_layers:
            if exists(attn):
                x = attn(x) + x

            x = net(x)
            layer_outputs.append(x)

        out = self.to_logits(x).flatten(1)

        img_32x32 = F.interpolate(orig_img, size = (32, 32))
        out_32x32 = self.to_shape_disc_out(img_32x32)

        if not calc_aux_loss:
            return out, out_32x32, None

        # self-supervised auto-encoding loss

        layer_8x8 = layer_outputs[-1]
        layer_16x16 = layer_outputs[-2]

        recon_img_8x8 = self.decoder1(layer_8x8)

        aux_loss = F.mse_loss(
            recon_img_8x8,
            F.interpolate(orig_img, size = recon_img_8x8.shape[2:])
        )

        if exists(self.decoder2):
            select_random_quadrant = lambda rand_quadrant, img: rearrange(img, 'b c (m h) (n w) -> (m n) b c h w', m = 2, n = 2)[rand_quadrant]
            crop_image_fn = partial(select_random_quadrant, floor(random() * 4))
            img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))

            recon_img_16x16 = self.decoder2(layer_16x16_part)

            aux_loss_16x16 = F.mse_loss(
                recon_img_16x16,
                F.interpolate(img_part, size = recon_img_16x16.shape[2:])
            )

            aux_loss = aux_loss + aux_loss_16x16

        return out, out_32x32, aux_loss


###############################################################################


class PainterLSTMSumGAN(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,
        num_layers,
        num_decoding_steps,
        out_tanh=False,
        use_pos_enc=False,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            num_layers = num_layers,
            num_decoding_steps = num_decoding_steps,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
        )

        self.G = LSTMSumGenerator(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = LSTMSumGenerator(**G_kwargs)
        set_requires_grad(self.GE, False)


        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(
                self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(
                m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


class PainterLSTMDeltaGAN(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,
        num_layers,
        num_decoding_steps,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        use_learned_init = False,
        val_tanh = False,
        vect_lr = False,
        use_intermed_loss = 1,
        scale_intermed_loss = True,
        wide_out_dim = 0,
        clip=0.0,
        clip_rnn=0.0,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            num_layers = num_layers,
            num_decoding_steps = num_decoding_steps,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
            use_softmax = use_softmax,
            no_sigmoid_lr = no_sigmoid_lr,
            val_tanh = val_tanh,
            vect_lr = vect_lr,
            use_learned_init = use_learned_init,
            use_intermed_loss = use_intermed_loss,
            scale_intermed_loss = scale_intermed_loss,
            wide_out_dim = wide_out_dim,
            clip = clip,
            clip_rnn = clip_rnn,
            rnn_init_fact = rnn_init_fact,
            kvb_init_fact = kvb_init_fact,
        )

        self.G = LSTMDeltaGenerator(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = LSTMDeltaGenerator(**G_kwargs)
        set_requires_grad(self.GE, False)

        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


class PainterRFWPDeltaGAN(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,
        num_layers,
        num_decoding_steps,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        use_learned_init = False,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            num_layers = num_layers,
            num_decoding_steps = num_decoding_steps,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
            use_softmax = use_softmax,
            no_sigmoid_lr = no_sigmoid_lr,
            use_learned_init = use_learned_init,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
        )

        self.G = RFWPDeltaGenerator(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = RFWPDeltaGenerator(**G_kwargs)
        set_requires_grad(self.GE, False)

        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


class PainterLSTMDeltaGANUnet(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,
        num_layers,
        num_decoding_steps,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        use_learned_init = False,
        clip=0.0,
        clip_rnn=0.0,
        stop_grad_unet = False,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        unet_lr = 2e-4,
        rank = 0,
        ddp = False,
        max_unet_dim = 4,
        frozen_base = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
        unet_tanh = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            num_layers = num_layers,
            num_decoding_steps = num_decoding_steps,
            stop_grad_unet = stop_grad_unet,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
            use_softmax = use_softmax,
            no_sigmoid_lr = no_sigmoid_lr,
            use_learned_init = use_learned_init,
            clip=clip,
            clip_rnn=clip_rnn,
            max_unet_dim=max_unet_dim,
            frozen_base=frozen_base,
            rnn_init_fact = rnn_init_fact,
            kvb_init_fact = kvb_init_fact,
            unet_tanh = unet_tanh,
        )

        self.G = LSTMDeltaGeneratorUnet(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = LSTMDeltaGeneratorUnet(**G_kwargs)
        set_requires_grad(self.GE, False)

        # separate learning rate for unet
        params_unet = {'params': self.G.mod_dict.unet.parameters(),
                       'lr': unet_lr}
        params_other = {'params': self.G.mod_dict.base.parameters()}
        opt_arg_params = [params_other, params_unet]

        if optimizer == "adam":
            self.G_opt = Adam(opt_arg_params, lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(
                self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self, only_unet=False):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        def update_moving_average_only_unet(ma_model, current_model):
            for current_name_params, ma_name_params in zip(current_model.named_parameters(), ma_model.named_parameters()):
                cur_name, current_params = current_name_params
                _, ma_params = ma_name_params

                if 'u_net' in cur_name:
                    old_weight, up_weight = ma_params.data, current_params.data
                    ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        if only_unet:
            update_moving_average_only_unet(self.GE, self.G)
        else:
            update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


# split latent
class PainterLSTMDeltaTimeSplitLatentGAN(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,  # not used
        num_layers,
        num_decoding_steps,
        out_tanh=False,
        use_pos_enc=False,
        use_softmax=False,
        no_sigmoid_lr = False,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            mini_latent_dim = mini_latent_dim,
            mini_input_size = mini_input_size,
            num_layers = num_layers,
            num_decoding_steps = num_decoding_steps,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
            use_softmax = use_softmax,
            no_sigmoid_lr = no_sigmoid_lr,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            rnn_init_fact = rnn_init_fact,
            kvb_init_fact = kvb_init_fact,
        )

        self.G = LSTMDeltaTimeSplitLatentGenerator(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = LSTMDeltaTimeSplitLatentGenerator(**G_kwargs)
        set_requires_grad(self.GE, False)


        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


class PainterLSTMDeltaTimeSplitLatentGANUnet(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,  # not used
        num_layers,
        num_decoding_steps,
        out_tanh=False,
        use_pos_enc=False,
        use_softmax=False,
        no_sigmoid_lr=False,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        stop_grad_unet = False,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False,
        max_unet_dim = 4,
        frozen_base = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            mini_latent_dim = mini_latent_dim,
            mini_input_size = mini_input_size,
            num_layers = num_layers,
            num_decoding_steps = num_decoding_steps,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
            use_softmax = use_softmax,
            no_sigmoid_lr = no_sigmoid_lr,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            stop_grad_unet = stop_grad_unet,
            max_unet_dim = max_unet_dim,
            frozen_base = frozen_base,
            rnn_init_fact = rnn_init_fact,
            kvb_init_fact = kvb_init_fact,
        )

        self.G = LSTMDeltaTimeSplitLatentGeneratorUnet(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = LSTMDeltaTimeSplitLatentGeneratorUnet(**G_kwargs)
        set_requires_grad(self.GE, False)


        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self, only_unet=False):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        def update_moving_average_only_unet(ma_model, current_model):
            for current_name_params, ma_name_params in zip(current_model.named_parameters(), ma_model.named_parameters()):
                cur_name, current_params = current_name_params
                _, ma_params = ma_name_params

                if 'u_net' in cur_name:
                    old_weight, up_weight = ma_params.data, current_params.data
                    ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        if only_unet:
            update_moving_average_only_unet(self.GE, self.G)
        else:
            update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


class PainterLSTMDeltaTimeSplitLatentV2GANUnet(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,  # not used
        num_layers,
        num_decoding_steps,
        out_tanh=False,
        use_pos_enc=False,
        use_softmax=False,
        no_sigmoid_lr=False,
        v2_input_sigmoid=False,
        val_tanh = False,
        vect_lr = False,
        use_latent_to_init = False,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        stop_grad_unet = False,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False,
        max_unet_dim = 4,
        frozen_base = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
        unet_tanh = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            mini_latent_dim = mini_latent_dim,
            mini_input_size = mini_input_size,
            v2_input_sigmoid = v2_input_sigmoid,
            val_tanh = val_tanh,
            num_layers = num_layers,
            vect_lr = vect_lr,
            use_latent_to_init = use_latent_to_init,
            num_decoding_steps = num_decoding_steps,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
            use_softmax = use_softmax,
            no_sigmoid_lr = no_sigmoid_lr,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            stop_grad_unet = stop_grad_unet,
            max_unet_dim = max_unet_dim,
            frozen_base = frozen_base,
            rnn_init_fact = rnn_init_fact,
            kvb_init_fact = kvb_init_fact,
            unet_tanh = unet_tanh,
        )

        self.G = LSTMDeltaTimeSplitLatentGeneratorV2Unet(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = LSTMDeltaTimeSplitLatentGeneratorV2Unet(**G_kwargs)
        set_requires_grad(self.GE, False)


        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self, only_unet=False):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        def update_moving_average_only_unet(ma_model, current_model):
            for current_name_params, ma_name_params in zip(current_model.named_parameters(), ma_model.named_parameters()):
                cur_name, current_params = current_name_params
                _, ma_params = ma_name_params

                if 'u_net' in cur_name:
                    old_weight, up_weight = ma_params.data, current_params.data
                    ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        if only_unet:
            update_moving_average_only_unet(self.GE, self.G)
        else:
            update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


# split latent
class PainterLSTMDeltaTimeSplitLatentGANv2(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,  # not used
        num_layers,
        num_decoding_steps,
        out_tanh=False,
        use_pos_enc=False,
        use_softmax=False,
        no_sigmoid_lr=False,
        v2_input_sigmoid=False,
        val_tanh=False,
        vect_lr=False,
        use_latent_to_init=False,
        use_intermed_loss=1,
        scale_intermed_loss=True,
        wide_out_dim=0,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            mini_latent_dim = mini_latent_dim,
            mini_input_size = mini_input_size,
            num_layers = num_layers,
            num_decoding_steps = num_decoding_steps,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
            use_softmax = use_softmax,
            no_sigmoid_lr = no_sigmoid_lr,
            v2_input_sigmoid = v2_input_sigmoid,
            val_tanh = val_tanh,
            vect_lr = vect_lr,
            use_latent_to_init = use_latent_to_init,
            use_intermed_loss = use_intermed_loss,
            scale_intermed_loss = scale_intermed_loss,
            wide_out_dim = wide_out_dim,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            rnn_init_fact = rnn_init_fact,
            kvb_init_fact = kvb_init_fact,
        )

        self.G = LSTMDeltaTimeSplitLatentGeneratorV2(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = LSTMDeltaTimeSplitLatentGeneratorV2(**G_kwargs)
        set_requires_grad(self.GE, False)


        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


# split latent
class PainterLSTMOjaTimeSplitLatentGANv2(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,  # not used
        num_layers,
        num_decoding_steps,
        out_tanh=False,
        use_pos_enc=False,
        use_softmax=False,
        no_sigmoid_lr=False,
        v2_input_sigmoid=False,
        val_tanh = False,
        vect_lr = False,
        use_intermed_loss=1,
        scale_intermed_loss=True,
        wide_out_dim=0,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        mini_latent_dim = 8,  # per time latent dim
        mini_input_size = 128,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            mini_latent_dim = mini_latent_dim,
            mini_input_size = mini_input_size,
            num_layers = num_layers,
            num_decoding_steps = num_decoding_steps,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
            use_softmax = use_softmax,
            no_sigmoid_lr = no_sigmoid_lr,
            v2_input_sigmoid = v2_input_sigmoid,
            val_tanh = val_tanh,
            vect_lr = vect_lr,
            use_intermed_loss = use_intermed_loss,
            scale_intermed_loss = scale_intermed_loss,
            wide_out_dim = wide_out_dim,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            rnn_init_fact = rnn_init_fact,
            kvb_init_fact = kvb_init_fact,
        )

        self.G = LSTMOjaTimeSplitLatentGeneratorV2(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = LSTMOjaTimeSplitLatentGeneratorV2(**G_kwargs)
        set_requires_grad(self.GE, False)


        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


# split latent
class PainterTransformerDeltaSplitLatentGAN(BaseModel):
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        rnn_hidden_size,  # not used
        num_layers,
        num_decoding_steps,
        out_tanh=False,
        use_pos_enc=False,
        use_softmax=False,
        no_sigmoid_lr=False,
        num_heads = 8,
        dim_feedforward = 1024,
        dropout = 0.0,
        mini_latent_dim = 8,  # per time latent dim
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            mini_latent_dim = mini_latent_dim,
            rnn_hidden_size = rnn_hidden_size,
            num_layers = num_layers,
            num_decoding_steps = num_decoding_steps,
            transparent = transparent,
            greyscale = greyscale,
            out_tanh = out_tanh,
            use_pos_enc = use_pos_enc,
            use_softmax = use_softmax,
            no_sigmoid_lr = no_sigmoid_lr,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
        )

        self.G = TransformerDeltaTimeSplitLatentGenerator(**G_kwargs)

        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        self.ema_updater = EMA(0.995)
        self.GE = TransformerDeltaTimeSplitLatentGenerator(**G_kwargs)
        set_requires_grad(self.GE, False)


        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        self.D.apply(self._init_weights)
        self.reset_parameter_averaging()

        self.cuda(rank)
        self.D_aug = AugWrapper(self.D, image_size)

    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        raise NotImplemented


###############################################################################


class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        optimizer = 'adam',
        save_best_and_last = True,
        num_workers = None,
        latent_dim = 256,
        image_size = 128,
        rnn_hidden_size = 512,
        num_layers = 1,
        num_heads = 8,
        dim_feedforward = 1024,
        mini_latent_dim = 8,
        mini_input_size = 128,
        wide_out_dim = 0,
        unet_lr = 2e-4,
        dropout = 0.0,
        clip = 0.0,
        clip_rnn = 0.0,
        max_unet_dim = 4,
        rnn_init_fact = 1,
        kvb_init_fact = 1,
        unet_tanh = False,
        frozen_base = False,
        num_decoding_steps = 64,
        out_tanh = False,
        use_pos_enc = False,
        use_softmax = False,
        no_sigmoid_lr = False,
        v2_input_sigmoid = False,
        val_tanh = False,
        vect_lr = False,
        use_latent_to_init = False,
        use_learned_init = False,
        use_pre_unet_loss = False,
        use_unet_match_loss = False,
        use_unet_match_loss_only = False,
        stop_grad_unet = False,
        use_intermed_loss = 1,
        scale_intermed_loss = True,
        model_type = 'base',
        num_image_tiles = 8,
        fmap_max = 512,
        transparent = False,
        greyscale = False,
        skip_resize = False,
        transform_to_greyscale = False,
        batch_size = 4,
        gp_weight = 10,
        gradient_accumulate_every = 1,
        attn_res_layers = [],
        freq_chan_attn = False,
        disc_output_size = 5,
        dual_contrast_loss = False,
        antialias = False,
        lr = 2e-4,
        lr_mlp = 1.,
        ttur_mult = 1.,
        save_every = 1000,
        evaluate_every = 1000,
        aug_prob = None,
        aug_types = ['translation', 'cutout'],
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        fid_all_real = True,
        fid_real_path = None,
        clear_fid_cache = False,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        amp = False,
        hparams = None,
        load_strict = True,
        use_wandb = False,
        loginf = None,
        image_save_format='jpg',
        ema_only_unet=False,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.image_save_format = image_save_format
        self.ema_only_unet = ema_only_unet

        self.use_pre_unet_loss = use_pre_unet_loss
        self.stop_grad_unet = stop_grad_unet
        self.use_unet_match_loss = use_unet_match_loss
        self.use_unet_match_loss_only = use_unet_match_loss_only

        self.transform_to_greyscale = transform_to_greyscale

        self.name = name
        self.model_type = model_type
        self.save_best_and_last = save_best_and_last
        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        assert fid_real_path is not None
        self.fid_real_path = Path(fid_real_path)
        self.frozen_base = frozen_base
        self.use_intermed_loss = use_intermed_loss
        assert num_decoding_steps % use_intermed_loss == 0
        self.scale_intermed_loss = scale_intermed_loss

        self.rnn_init_fact = rnn_init_fact
        self.kvb_init_fact = kvb_init_fact

        self.unet_tanh = unet_tanh

        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
        self.loginf = loginf

        self.config_path = self.models_dir / name / '.config.json'

        assert is_power_of_two(image_size), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        assert all(map(is_power_of_two, attn_res_layers)), 'resolution layers of attention must all be powers of 2 (16, 32, 64, 128, 256, 512)'

        assert not (dual_contrast_loss and disc_output_size > 1), 'discriminator output size cannot be greater than 1 if using dual contrastive loss'

        self.image_size = image_size
        self.num_image_tiles = num_image_tiles

        self.latent_dim = latent_dim
        self.fmap_max = fmap_max
        self.transparent = transparent
        self.greyscale = greyscale
        self.skip_resize = skip_resize

        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_decoding_steps = num_decoding_steps
        self.out_tanh = out_tanh
        self.use_pos_enc = use_pos_enc
        self.use_softmax = use_softmax
        self.no_sigmoid_lr = no_sigmoid_lr
        self.v2_input_sigmoid = v2_input_sigmoid
        self.val_tanh = val_tanh
        self.vect_lr = vect_lr
        self.use_latent_to_init = use_latent_to_init

        self.use_learned_init = use_learned_init
        self.max_unet_dim = max_unet_dim

        self.mini_latent_dim = mini_latent_dim
        self.mini_input_size = mini_input_size
        self.wide_out_dim = wide_out_dim

        self.clip = clip
        self.clip_rnn = clip_rnn

        assert (int(self.transparent) + int(self.greyscale)) < 2, 'you can only set either transparency or greyscale'

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.unet_lr = unet_lr
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.gp_weight = gp_weight

        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.attn_res_layers = attn_res_layers
        self.freq_chan_attn = freq_chan_attn

        self.disc_output_size = disc_output_size
        self.antialias = antialias

        self.dual_contrast_loss = dual_contrast_loss

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None
        self.best_fid = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.fid_all_real = fid_all_real
        self.clear_fid_cache = clear_fid_cache

        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.syncbatchnorm = is_ddp

        self.load_strict = load_strict

        self.amp = amp
        self.G_scaler = GradScaler(enabled = self.amp)
        self.D_scaler = GradScaler(enabled = self.amp)

    @property
    def image_extension(self):
        return self.image_save_format

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
        
    def init_GAN(self):
        args, kwargs = self.GAN_params

        # set some global variables before instantiating GAN

        global norm_class
        global Blur

        norm_class = nn.SyncBatchNorm if self.syncbatchnorm else nn.BatchNorm2d
        Blur = nn.Identity if not self.antialias else Blur

        # handle bugs when
        # switching from multi-gpu back to single gpu

        if self.syncbatchnorm and not self.is_ddp:
            import torch.distributed as dist
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group('nccl', rank=0, world_size=1)

        # instantiate GAN

        if self.model_type == 'lstm_sum':
            self.GAN = PainterLSTMSumGAN(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                num_layers = self.num_layers,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                *args,
                **kwargs
            )
        elif self.model_type == 'trafo_delta_split':
            self.GAN = PainterTransformerDeltaSplitLatentGAN(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                mini_latent_dim = self.mini_latent_dim,
                num_layers = self.num_layers,
                num_heads = self.num_heads,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                use_softmax=self.use_softmax,
                no_sigmoid_lr=self.no_sigmoid_lr,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                *args,
                **kwargs
            )
        elif self.model_type == 'lstm_delta_split':
            self.GAN = PainterLSTMDeltaTimeSplitLatentGAN(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                mini_latent_dim = self.mini_latent_dim,
                mini_input_size = self.mini_input_size,
                num_layers = self.num_layers,
                num_heads = self.num_heads,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                use_softmax=self.use_softmax,
                no_sigmoid_lr=self.no_sigmoid_lr,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                rnn_init_fact = self.rnn_init_fact,
                kvb_init_fact = self.kvb_init_fact,
                *args,
                **kwargs
            )
        elif self.model_type == 'lstm_delta_split_v2':
            self.GAN = PainterLSTMDeltaTimeSplitLatentGANv2(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                mini_latent_dim = self.mini_latent_dim,
                mini_input_size = self.mini_input_size,
                num_layers = self.num_layers,
                num_heads = self.num_heads,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                use_softmax=self.use_softmax,
                no_sigmoid_lr=self.no_sigmoid_lr,
                v2_input_sigmoid=self.v2_input_sigmoid,
                val_tanh=self.val_tanh,
                vect_lr=self.vect_lr,
                use_latent_to_init=self.use_latent_to_init,
                use_intermed_loss=self.use_intermed_loss,
                scale_intermed_loss=self.scale_intermed_loss,
                wide_out_dim=self.wide_out_dim,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                rnn_init_fact = self.rnn_init_fact,
                kvb_init_fact = self.kvb_init_fact,
                *args,
                **kwargs
            )
        elif self.model_type == 'lstm_oja_split_v2':
            self.GAN = PainterLSTMOjaTimeSplitLatentGANv2(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                mini_latent_dim = self.mini_latent_dim,
                mini_input_size = self.mini_input_size,
                num_layers = self.num_layers,
                num_heads = self.num_heads,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                use_softmax=self.use_softmax,
                no_sigmoid_lr=self.no_sigmoid_lr,
                v2_input_sigmoid=self.v2_input_sigmoid,
                val_tanh=self.val_tanh,
                vect_lr=self.vect_lr,
                use_intermed_loss=self.use_intermed_loss,
                scale_intermed_loss=self.scale_intermed_loss,
                wide_out_dim=self.wide_out_dim,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                rnn_init_fact = self.rnn_init_fact,
                kvb_init_fact = self.kvb_init_fact,
                *args,
                **kwargs
            )
        # different latent for each position
        elif self.model_type == 'lstm_delta':
            self.GAN = PainterLSTMDeltaGAN(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                num_layers = self.num_layers,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                use_softmax=self.use_softmax,
                no_sigmoid_lr=self.no_sigmoid_lr,
                val_tanh=self.val_tanh,
                vect_lr=self.vect_lr,
                use_learned_init=self.use_learned_init,
                use_intermed_loss=self.use_intermed_loss,
                scale_intermed_loss=self.scale_intermed_loss,
                wide_out_dim=self.wide_out_dim,
                clip=self.clip,
                clip_rnn=self.clip_rnn,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                rnn_init_fact = self.rnn_init_fact,
                kvb_init_fact = self.kvb_init_fact,
                *args,
                **kwargs
            )
        elif self.model_type == 'rfwp_delta':
            self.GAN = PainterRFWPDeltaGAN(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                num_layers = self.num_layers,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                use_softmax=self.use_softmax,
                no_sigmoid_lr=self.no_sigmoid_lr,
                use_learned_init=self.use_learned_init,
                num_heads = self.num_heads,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                *args,
                **kwargs
            )
        elif self.model_type == 'lstm_delta_unet':
            self.GAN = PainterLSTMDeltaGANUnet(
                optimizer=self.optimizer,
                lr = self.lr,
                unet_lr = self.unet_lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                num_layers = self.num_layers,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                use_softmax=self.use_softmax,
                no_sigmoid_lr=self.no_sigmoid_lr,
                use_learned_init=self.use_learned_init,
                clip=self.clip,
                clip_rnn=self.clip_rnn,
                max_unet_dim=self.max_unet_dim,
                stop_grad_unet=self.stop_grad_unet,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                frozen_base = self.frozen_base,
                rnn_init_fact = self.rnn_init_fact,
                kvb_init_fact = self.kvb_init_fact,
                unet_tanh = self.unet_tanh,
                *args,
                **kwargs
            )
        elif self.model_type == 'lstm_delta_split_unet':
            self.GAN = PainterLSTMDeltaTimeSplitLatentGANUnet(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                mini_latent_dim = self.mini_latent_dim,
                mini_input_size = self.mini_input_size,
                num_layers = self.num_layers,
                num_heads = self.num_heads,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                use_softmax=self.use_softmax,
                no_sigmoid_lr=self.no_sigmoid_lr,
                max_unet_dim=self.max_unet_dim,
                stop_grad_unet=self.stop_grad_unet,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                frozen_base = self.frozen_base,
                rnn_init_fact = self.rnn_init_fact,
                kvb_init_fact = self.kvb_init_fact,
                *args,
                **kwargs
            )
        elif self.model_type == 'lstm_delta_split_v2_unet':
            self.GAN = PainterLSTMDeltaTimeSplitLatentV2GANUnet(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                rnn_hidden_size = self.rnn_hidden_size,
                mini_latent_dim = self.mini_latent_dim,
                mini_input_size = self.mini_input_size,
                num_layers = self.num_layers,
                num_heads = self.num_heads,
                dim_feedforward = self.dim_feedforward,
                dropout = self.dropout,
                num_decoding_steps = self.num_decoding_steps,
                out_tanh=self.out_tanh,
                use_pos_enc=self.use_pos_enc,
                use_softmax=self.use_softmax,
                no_sigmoid_lr=self.no_sigmoid_lr,
                v2_input_sigmoid=self.v2_input_sigmoid,
                val_tanh=self.val_tanh,
                vect_lr=self.vect_lr,
                use_latent_to_init=self.use_latent_to_init,
                max_unet_dim=self.max_unet_dim,
                stop_grad_unet=self.stop_grad_unet,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                frozen_base = self.frozen_base,
                rnn_init_fact = self.rnn_init_fact,
                kvb_init_fact = self.kvb_init_fact,
                unet_tanh = self.unet_tanh,
                *args,
                **kwargs
            )
        else:
            self.GAN = LightweightGAN(
                optimizer=self.optimizer,
                lr = self.lr,
                latent_dim = self.latent_dim,
                attn_res_layers = self.attn_res_layers,
                freq_chan_attn = self.freq_chan_attn,
                image_size = self.image_size,
                ttur_mult = self.ttur_mult,
                fmap_max = self.fmap_max,
                disc_output_size = self.disc_output_size,
                transparent = self.transparent,
                greyscale = self.greyscale,
                rank = self.rank,
                *args,
                **kwargs
            )
        if self.loginf is not None:
            self.loginf(f"Num params, all: {self.GAN.num_params()}")
            self.loginf(f"Num params, generator: {self.GAN.G.num_params()}")
            self.loginf(f"Num params, discriminator: {self.GAN.D.num_params()}")

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank], 'output_device': self.rank, 'find_unused_parameters': True}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.transparent = config['transparent']
        self.syncbatchnorm = config['syncbatchnorm']
        self.disc_output_size = config['disc_output_size']
        self.greyscale = config.pop('greyscale', False)
        self.attn_res_layers = config.pop('attn_res_layers', [])
        self.freq_chan_attn = config.pop('freq_chan_attn', False)
        self.optimizer = config.pop('optimizer', 'adam')
        self.fmap_max = config.pop('fmap_max', 512)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            'image_size': self.image_size,
            'transparent': self.transparent,
            'greyscale': self.greyscale,
            'syncbatchnorm': self.syncbatchnorm,
            'disc_output_size': self.disc_output_size,
            'optimizer': self.optimizer,
            'attn_res_layers': self.attn_res_layers,
            'freq_chan_attn': self.freq_chan_attn
        }

    def set_data_src(self, folder):
        num_workers = default(self.num_workers, math.ceil(NUM_CORES / self.world_size))
        if self.transform_to_greyscale:  # we only need false here, in other places self.greyscale is used to specify number of channels in models
            dataset_greyscale = False
        else:
            dataset_greyscale = self.greyscale
        self.dataset = ImageDataset(
            folder, self.image_size, transparent = self.transparent,
            greyscale = dataset_greyscale, skip_resize = self.skip_resize,
            aug_prob = self.dataset_aug_prob,
            transform_to_greyscale = self.transform_to_greyscale)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # If FID is computed, all real images are used, but images are not there yet
        if self.calculate_fid_every is not None and self.fid_all_real:
                if not self.fid_real_path.exists() or self.clear_fid_cache:
                    rmtree(self.fid_real_path, ignore_errors=True)
                    os.makedirs(self.fid_real_path)

                    dummy_loader = DataLoader(
                        self.dataset, num_workers = num_workers,
                        batch_size = self.batch_size,
                        shuffle = False, drop_last = False,
                        pin_memory = False)
                    batch_num = 0
                    print(f'Saving resized real images for FID: {self.fid_real_path}')
                    with tqdm(total=len(dummy_loader)) as progress_bar:
                        for real_resized_image in dummy_loader:
                            for k, image in enumerate(real_resized_image.unbind(0)):
                                ind = k + batch_num * self.batch_size
                                torchvision.utils.save_image(
                                    image,
                                    self.fid_real_path / f'{ind}.{self.image_save_format}')
                            progress_bar.update(1)
                            batch_num += 1
                    del dummy_loader
                else:
                    print(f'Using existing real images: {self.fid_real_path}')

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'
        device = torch.device(f'cuda:{self.rank}')

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.zeros([], device=device)
        total_gen_loss = torch.zeros([], device=device)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.image_size
        latent_dim = self.GAN.latent_dim

        aug_prob   = default(self.aug_prob, 0)
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        apply_gradient_penalty = self.steps % 4 == 0

        # amp related contexts and functions

        amp_context = autocast if self.amp else null_context

        # discriminator loss fn

        if self.dual_contrast_loss:
            D_loss_fn = dual_contrastive_loss
        elif self.model_type in ['lstm_delta_unet', 'lstm_delta_split_unet', 'lstm_delta_split_v2_unet']:
            D_loss_fn = hinge_loss_bis
        elif self.model_type in ['lstm_delta', 'lstm_delta_split_v2', 'lstm_oja_split_v2'] and self.use_intermed_loss > 1:
            D_loss_fn = hinge_loss_bis
        else:
            D_loss_fn = hinge_loss

        # train discriminator

        self.GAN.D_opt.zero_grad()
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, G]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)
            image_batch = next(self.loader).cuda(self.rank)
            image_batch.requires_grad_()

            with amp_context():
                with torch.no_grad():
                    if self.model_type in ['lstm_delta_unet', 'lstm_delta_split_unet', 'lstm_delta_split_v2_unet']:
                        generated_images = G(latents, self.use_pre_unet_loss)
                    elif self.model_type in ['lstm_delta', 'lstm_delta_split_v2', 'lstm_oja_split_v2'] and self.use_intermed_loss > 1:
                        generated_images = G(latents, output_only_last=False)
                    else:
                        generated_images = G(latents)

                if self.use_pre_unet_loss and self.use_unet_match_loss_only:
                    _, generated_images = torch.split(generated_images, [self.batch_size, self.batch_size], dim=0)

                fake_output, fake_output_32x32, _ = D_aug(generated_images, detach = True, **aug_kwargs)
                real_output, real_output_32x32, real_aux_loss = D_aug(image_batch,  calc_aux_loss = True, **aug_kwargs)

                real_output_loss = real_output
                fake_output_loss = fake_output

                divergence = D_loss_fn(real_output_loss, fake_output_loss)
                divergence_32x32 = D_loss_fn(real_output_32x32, fake_output_32x32)
                disc_loss = divergence + divergence_32x32

                aux_loss = real_aux_loss
                disc_loss = disc_loss + aux_loss

            if apply_gradient_penalty:
                outputs = [real_output, real_output_32x32]
                outputs = list(map(self.D_scaler.scale, outputs)) if self.amp else outputs

                scaled_gradients = torch_grad(outputs=outputs, inputs=image_batch,
                                       grad_outputs=list(map(lambda t: torch.ones(t.size(), device = image_batch.device), outputs)),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0]

                inv_scale = safe_div(1., self.D_scaler.get_scale()) if self.amp else 1.

                if inv_scale != float('inf'):
                    gradients = scaled_gradients * inv_scale

                    with amp_context():
                        gradients = gradients.reshape(batch_size, -1)
                        gp =  self.gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                        if not torch.isnan(gp):
                            disc_loss = disc_loss + gp
                            self.last_gp_loss = gp.clone().detach().item()

            with amp_context():
                disc_loss = disc_loss / self.gradient_accumulate_every

            disc_loss.register_hook(raise_if_nan)
            self.D_scaler.scale(disc_loss).backward()
            total_disc_loss += divergence

        self.last_recon_loss = aux_loss.item()
        self.d_loss = float(total_disc_loss.item() / self.gradient_accumulate_every)
        self.D_scaler.step(self.GAN.D_opt)
        self.D_scaler.update()

        # generator loss fn

        if self.dual_contrast_loss:
            G_loss_fn = dual_contrastive_loss
            G_requires_calc_real = True
        else:
            G_loss_fn = gen_ce_loss
            G_requires_calc_real = False

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[G, D_aug]):
            latents = torch.randn(batch_size, latent_dim).cuda(self.rank)

            if G_requires_calc_real:
                image_batch = next(self.loader).cuda(self.rank)
                image_batch.requires_grad_()

            with amp_context():
                if self.model_type in ['lstm_delta_unet', 'lstm_delta_split_unet', 'lstm_delta_split_v2_unet']:
                    assert not G_requires_calc_real
                    generated_images = G(latents, self.use_pre_unet_loss)
                elif self.model_type in ['lstm_delta', 'lstm_delta_split_v2', 'lstm_oja_split_v2'] and self.use_intermed_loss > 1:
                    generated_images = G(latents, output_only_last=False)
                else:
                    generated_images = G(latents)

                if self.use_pre_unet_loss and self.use_unet_match_loss_only:
                    assert self.model_type in ['lstm_delta_unet', 'lstm_delta_split_unet', 'lstm_delta_split_v2_unet']
                    assert self.use_pre_unet_loss
                    fwp_out, generated_images = torch.split(generated_images, [self.batch_size, self.batch_size], dim=0)

                fake_output, fake_output_32x32, _ = D_aug(generated_images, **aug_kwargs)
                real_output, real_output_32x32, _ = D_aug(image_batch, **aug_kwargs) if G_requires_calc_real else (None, None, None)

                loss = G_loss_fn(fake_output, real_output)
                loss_32x32 = G_loss_fn(fake_output_32x32, real_output_32x32)

                gen_loss = loss + loss_32x32

                if self.use_unet_match_loss:
                    assert self.model_type in ['lstm_delta_unet', 'lstm_delta_split_unet', 'lstm_delta_split_v2_unet']
                    assert self.use_pre_unet_loss
                    if not self.use_unet_match_loss_only:
                        fwp_out, generated_images = torch.split(generated_images, [self.batch_size, self.batch_size], dim=0)
                    mse_loss_fn = nn.MSELoss()
                    gen_loss = gen_loss + mse_loss_fn(generated_images.detach(), fwp_out)

                gen_loss = gen_loss / self.gradient_accumulate_every

            gen_loss.register_hook(raise_if_nan)
            self.G_scaler.scale(gen_loss).backward()
            total_gen_loss += loss

        self.g_loss = float(total_gen_loss.item() / self.gradient_accumulate_every)
        # apply clipping
        if self.model_type in ['lstm_delta', 'lstm_delta_unet']:
            if self.GAN.G.clip > 0.0:  # apply global clipping in priority
                nn.utils.clip_grad_norm_(self.GAN.G.parameters(), self.GAN.G.clip)
            elif self.GAN.G.clip_rnn > 0.0:
                nn.utils.clip_grad_norm_(self.GAN.G.decoder_rnn.parameters(), self.GAN.G.clip_rnn)
        self.G_scaler.step(self.GAN.G_opt)
        self.G_scaler.update()

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            if 'unet' in self.model_type:
                self.GAN.EMA(only_unet=self.ema_only_unet)
            else:
                self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from latest checkpoint #{0}')
            self.load(0)
            raise NanException

        del total_disc_loss
        del total_gen_loss

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                if self.save_best_and_last:  # overwrite the latest
                    self.save(0)
                else:
                    self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 20000):
                self.evaluate(floor(self.steps / self.evaluate_every), num_image_tiles = self.num_image_tiles)

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid
                if self.best_fid is not None:
                    if fid < self.best_fid:
                        self.best_fid = fid
                        if self.save_best_and_last:
                            self.save(1)
                else:
                    self.best_fid = fid


                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 4):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles
    
        latent_dim = self.GAN.latent_dim

        # latents and noise
        def image_to_pil(image):
            ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            return im

        latents = det_randn((num_rows ** 2, latent_dim)).cuda(self.rank)
        interpolate_latents = interpolate_between(latents[:num_rows], latents[-num_rows:],
                                                  num_samples=num_rows,
                                                  dim=0).flatten(end_dim=1)

        if self.model_type in ['lstm_delta_unet', 'lstm_delta_split_unet', 'lstm_delta_split_v2_unet']:
            self.GAN.GE.output = 'post_unet'
            generate_interpolations = self.generate_(self.GAN.GE, interpolate_latents)
            torchvision.utils.save_image(
                generate_interpolations,
                str(self.results_dir / self.name / f'{str(num)}-post_interp.{ext}'),
                nrow=num_rows)
            if self.use_wandb:
                import wandb
                wandb.log({"post_interpolation" : wandb.Image(generate_interpolations)})
            self.GAN.GE.output = 'pre_unet'
            generate_interpolations = self.generate_(self.GAN.GE, interpolate_latents)
            torchvision.utils.save_image(
                generate_interpolations,
                str(self.results_dir / self.name / f'{str(num)}-pre_interp.{ext}'),
                nrow=num_rows)
            if self.use_wandb:
                wandb.log({"pre_interpolation" : wandb.Image(generate_interpolations)})
            self.GAN.GE.output = 'not_specified'
        else:
            generate_interpolations = self.generate_(self.GAN.GE, interpolate_latents)
            torchvision.utils.save_image(
                generate_interpolations,
                str(self.results_dir / self.name / f'{str(num)}-interp.{ext}'),
                nrow=num_rows)
            if self.use_wandb:
                import wandb
                wandb.log({"interpolation" : wandb.Image(generate_interpolations)})

        # regular
        if self.model_type in ['lstm_delta_unet', 'lstm_delta_split_unet', 'lstm_delta_split_v2_unet']:
            self.GAN.G.output = 'post_unet'
            generated_images = self.generate_(self.GAN.G, latents)
            torchvision.utils.save_image(
                generated_images,
                str(self.results_dir / self.name / f'post-{str(num)}.{ext}'),
                nrow=num_rows)

            if self.use_wandb:
                wandb.log({"post_unet_checkpoint" : wandb.Image(generated_images)})

            self.GAN.G.output = 'pre_unet'
            generated_images = self.generate_(self.GAN.G, latents)
            torchvision.utils.save_image(
                generated_images,
                str(self.results_dir / self.name / f'pre-{str(num)}.{ext}'),
                nrow=num_rows)

            if self.use_wandb:
                wandb.log({"pre_unet_checkpoint" : wandb.Image(generated_images)})

            self.GAN.G.output = 'not_specified'

        else:
            generated_images = self.generate_(self.GAN.G, latents)
            torchvision.utils.save_image(
                generated_images,
                str(self.results_dir / self.name / f'{str(num)}.{ext}'),
                nrow=num_rows)

            if self.use_wandb:
                wandb.log({"checkpoint" : wandb.Image(generated_images)})

        # moving averages

        if self.model_type in ['lstm_delta_unet', 'lstm_delta_split_unet', 'lstm_delta_split_v2_unet']:
            self.GAN.GE.output = 'post_unet'
            generated_images = self.generate_(self.GAN.GE, latents)
            torchvision.utils.save_image(
                generated_images,
                str(self.results_dir / self.name / f'{str(num)}-post-ema.{ext}'),
                nrow=num_rows)
            if self.use_wandb:
                wandb.log({"post_unet_ema" : wandb.Image(generated_images)})

            self.GAN.GE.output = 'pre_unet'
            generated_images = self.generate_(self.GAN.GE, latents)
            torchvision.utils.save_image(
                generated_images,
                str(self.results_dir / self.name / f'{str(num)}-pre-ema.{ext}'),
                nrow=num_rows)
            if self.use_wandb:
                wandb.log({"pre_unet_ema" : wandb.Image(generated_images)})
            self.GAN.GE.output = 'not_specified'
        else:
            generated_images = self.generate_(self.GAN.GE, latents)
            torchvision.utils.save_image(
                generated_images,
                str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'),
                nrow=num_rows)
            if self.use_wandb:
                wandb.log({"ema" : wandb.Image(generated_images)})

    @torch.no_grad()
    def generate(self, num=0, num_image_tiles=4, checkpoint=None, types=['default', 'ema']):
        self.GAN.eval()

        latent_dim = self.GAN.latent_dim
        dir_name = self.name + str('-generated-') + str(checkpoint)
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension

        if not dir_full.exists():
            os.mkdir(dir_full)

        # regular
        if 'default' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated default images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-default.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        # moving averages
        if 'ema' in types:
            import time
            start_time = time.time()
            for i in tqdm(range(num_image_tiles), desc='Saving generated EMA images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-ema.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)
            elapsed = time.time() - start_time
            print(f"Elapsed {elapsed / 60.:.2f} min. Processed {num_image_tiles / elapsed:.1f} images per second.")

        return dir_full

    @torch.no_grad()
    def generate_steps(self, num=0, num_image_tiles=4, checkpoint=None, types=['default', 'ema'], num_rows=None):
        self.GAN.eval()

        if num_rows is None:
            num_rows = self.num_decoding_steps
            # num_rows = 16

        latent_dim = self.GAN.latent_dim
        dir_name = self.name + str('-generated-') + str(checkpoint)
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension

        if not dir_full.exists():
            os.mkdir(dir_full)

        # regular
        if 'default' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated Non-EMA images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.G, latents)
                generated_image = torchvision.utils.make_grid(generated_image, nrow=num_rows)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-default.{ext}')
                torchvision.utils.save_image(generated_image, path, nrow=1)

        # moving averages
        if 'ema' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated EMA images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.GE, latents)
                generated_image = torchvision.utils.make_grid(generated_image, nrow=num_rows, padding=2, pad_value=0.5)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-ema.{ext}')
                torchvision.utils.save_image(generated_image, path, nrow=1)

        return dir_full

    @torch.no_grad()
    def show_progress(self, num_images=4, types=['default', 'ema']):
        checkpoints = self.get_checkpoints()
        assert exists(checkpoints), 'cannot find any checkpoints to create a training progress video for'

        dir_name = self.name + str('-progress')
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension
        latents = None

        zfill_length = math.ceil(math.log10(len(checkpoints)))

        if not dir_full.exists():
            os.mkdir(dir_full)

        for checkpoint in tqdm(checkpoints, desc='Generating progress images'):
            self.load(checkpoint, print_version=False)
            self.GAN.eval()

            if checkpoint == 0:
                latents = torch.randn((num_images, self.GAN.latent_dim)).cuda(self.rank)

            # regular
            if 'default' in types:
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}.{ext}')
                torchvision.utils.save_image(generated_image, path, nrow=num_images)

            # moving averages
            if 'ema' in types:
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}-ema.{ext}')
                torchvision.utils.save_image(generated_image, path, nrow=num_images)

    @torch.no_grad()
    def calculate_fid(self, num_batches, eval_without_ema=False, report_time=False):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = self.fid_real_path
        fake_path = self.fid_dir / 'fake'

        if not self.fid_all_real:
            # remove any existing files used for fid calculation and recreate directories
            real_path = self.fid_dir / 'real'
            if not real_path.exists() or self.clear_fid_cache:
                rmtree(real_path, ignore_errors=True)
                os.makedirs(real_path)

                for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                    real_batch = next(self.loader)
                    for k, image in enumerate(real_batch.unbind(0)):
                        ind = k + batch_num * self.batch_size
                        torchvision.utils.save_image(image, real_path / f'{ind}.{self.image_save_format}')

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.latent_dim

        if report_time:
            import time
            start_time = time.time()
        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = torch.randn(self.batch_size, latent_dim).cuda(self.rank)

            # moving averages
            if eval_without_ema:
                generated_images = self.generate_(self.GAN.G, latents)
            else:
                generated_images = self.generate_(self.GAN.GE, latents)

            for j, image in enumerate(generated_images.unbind(0)):
                ind = j + batch_num * self.batch_size
                torchvision.utils.save_image(image, str(fake_path / f'{str(ind)}-ema.{ext}'))
        if report_time:
            elapsed = time.time() - start_time
            print(f"Elapsed {elapsed / 60.:.2f} min. Processed {self.calculate_fid_num_images / elapsed:.1f} images per second.")

        # This is inefficient as the same stats for real images are computed again everytime
        # Should better save the real stats
        score = fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, latents.device, 2048)
        # clean up
        rmtree(fake_path, ignore_errors=True)

        return score

    @torch.no_grad()
    def generate_(self, G, style, num_image_tiles = 8):
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim

        # latents and noise

        latents_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
        latents_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            generated_images = self.generate_(self.GAN.GE, interp_latents)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new('RGBA', pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self, loginf, use_wandb=False):
        if use_wandb:
            import wandb
            wandb.log({
                "G_loss": self.g_loss,
                "D_loss": self.d_loss,
                "scaled_gradient_l2": self.last_gp_loss,
                "D_aux_loss": self.last_recon_loss,
                'FID': self.last_fid,
            })

        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('SS', self.last_recon_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]

        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        # print(log)
        loginf(log)

        if self.run is not None:
            for key, value in data:
                self.run.track(value, key, step=self.steps)

        return data

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__,
            'G_scaler': self.G_scaler.state_dict(),
            'D_scaler': self.D_scaler.state_dict(),
            'G_opt': self.GAN.G_opt.state_dict(),
            'D_opt': self.GAN.D_opt.state_dict(),
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1, print_version=True, g_load_optimizer=True,
             d_load_optimizer=True, loginf=None):
        self.load_config()
        assert loginf is not None

        name = num
        if num == -1:
            checkpoints = self.get_checkpoints()

            if not exists(checkpoints):
                return

            name = checkpoints[-1]
            loginf(f'continuing from previous epoch - {name}')
            loginf(f'Resetting G optimizer: {not g_load_optimizer}')
            loginf(f'Resetting D optimizer: {not d_load_optimizer}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if print_version and 'version' in load_data and self.is_main:
            loginf(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'], strict = self.load_strict)
        except Exception as e:
            saved_version = load_data['version']
            loginf('unable to load save model. please try downgrading the package to the version specified by the saved model (to do so, just run `pip install lightweight-gan=={saved_version}`')
            raise e

        if g_load_optimizer and 'G_opt' in load_data:
            self.GAN.G_opt.load_state_dict(load_data['G_opt'])
            self.G_scaler.load_state_dict(load_data['G_scaler'])
        if d_load_optimizer and 'D_opt' in load_data:
            self.GAN.D_opt.load_state_dict(load_data['D_opt'])
            self.D_scaler.load_state_dict(load_data['D_scaler'])

    def get_checkpoints(self):
        file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
        saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))

        if len(saved_nums) == 0:
            return None

        return saved_nums
