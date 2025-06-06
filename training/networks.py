# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        beta_d          = 19.9,         # Extent of the noise level schedule.
        beta_min        = 0.1,          # Initial slope of the noise level schedule.
        M               = 1000,         # Original number of timesteps in the DDPM formulation.
        epsilon_t       = 1e-5,         # Minimum t-value used during training.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VEPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        sigma_min       = 0.02,         # Minimum supported noise level.
        sigma_max       = 100,          # Maximum supported noise level.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import re
from dataclasses import dataclass
from typing import ClassVar
import random
import numpy as np
torch.manual_seed(46)
random.seed(46)
np.random.seed(46)

class ResnetBlock(nn.Module):

    def __init__(self, input_channels, output_channels, time_embeddings, num_groups, eps):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_embeddings = time_embeddings
        self.num_groups = num_groups
        self.eps = eps

        self.norm1 = nn.GroupNorm(num_groups=self.num_groups,
                            num_channels=self.input_channels,
                            eps=self.eps)
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels,
                            kernel_size=3, stride=1, padding=1, bias=True)
        self.nonlinearity = nn.SiLU()
        self.time_emb_proj = nn.Linear(in_features=self.time_embeddings,
                                            out_features=self.output_channels, bias=True)
        self.norm2 = nn.GroupNorm(num_groups=self.num_groups,
                            num_channels=self.output_channels,
                            eps=self.eps)
        self.conv2 = nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels,
                            kernel_size=3, stride=1, padding=1, bias=True)

        use_conv_shortcut = True if input_channels != output_channels else False
        self.conv_shortcut = None
        if use_conv_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels=self.input_channels,
                                                out_channels=self.output_channels, kernel_size=1,
                                                stride=1, padding=0, bias=True)

    def forward(self, x, temb):
        hidden_states = x # Start with input for residual connection

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        hidden_states = hidden_states + x

        return hidden_states

class UpsamplerBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                        kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984()
        scale_factor = 2.0
        if x.numel() * scale_factor > pow(2, 31) or x.shape[0] >= 64:
            x = x.contiguous()
        x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        x = self.conv(x)

        return x

class TimeEmbeddings(nn.Module):
    """
    Calculates sinusoidal embeddings and projects them.
    Matches diffusers Timesteps + TimestepEmbedding structure.
    """
    def __init__(self,
                 sinusoidal_dim: int,
                 output_dim: int,
                 max_period=10000):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.output_dim = output_dim
        if sinusoidal_dim % 2 != 0:
            raise ValueError(
                f"Cannot use sinusoidal dim {sinusoidal_dim}, "
                f"must be even."
            )
        half_dim = sinusoidal_dim // 2
        # we need to include the device in the init as this precomputations only
        # work without error in cuda
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device="cuda"
        )
        exponent = exponent / half_dim
        # Store as 'inv_freq' (inverse frequencies scaled)
        # Shape: [half_dim]
        self.register_buffer(
            'inv_freq', torch.exp(exponent), persistent=False
        )

        # Layers for projection, matching diffusers' TimestepEmbedding
        self.linear_1 = nn.Linear(sinusoidal_dim, output_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def _get_sinusoidal_embeddings(self, timesteps: torch.Tensor):
        """Calculates the base sinusoidal embeddings."""
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

        # Output of multiplication: [batch_size, half_dim]
        emb = timesteps[:, None].float() * self.inv_freq[None, :]

        # concat sine and cosine embeddings
        # Shape: [batch_size, sinusoidal_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Optional: Flip sin and cosine (matches provided code)
        # Note: Standard diffusers might just concat sin(first half)
        # and cos(second half) differently. Verify if exact
        # replication is needed.
        half_dim = self.sinusoidal_dim // 2
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)


        # print(f"Check emb flip cosine {torch.norm(emb-emb_flip)}") huge
        # Zero pad if sinusoidal_dim is odd
        if self.sinusoidal_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb

    def forward(self, timesteps: torch.Tensor, sample:torch.Tensor):
        # 1. Calculate sinusoidal embeddings
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        sin_emb = self._get_sinusoidal_embeddings(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        sin_emb = sin_emb.to(dtype=sample.dtype)


        # 2. Project embeddings (matches diffusers TimestepEmbedding)
        emb = self.linear_1(sin_emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)
        return emb

class Attention(nn.Module):

    def __init__(self, input_channels, n_head, cross_attention_dim=None,
                 qv_norm:str=None):
        super().__init__()
        self.input_channels = input_channels
        self.n_head = n_head
        self.cross_attention_dim = input_channels
        if cross_attention_dim:
            self.cross_attention_dim = cross_attention_dim

        dim_head = input_channels//self.n_head
        if qv_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(dim_head, eps=1e-6)
            # k has the same C as Q and V because the to_k and to_v
            self.norm_k = nn.RMSNorm(dim_head, eps=1e-6)
        else:
          self.norm_q = None
          self.norm_k = None

        self.to_q = nn.Linear(self.input_channels, self.input_channels, bias=False)
        self.to_k = nn.Linear(self.cross_attention_dim, self.input_channels, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.input_channels, bias=False)

        self.to_out = nn.Linear(self.input_channels, self.input_channels, bias=True)

    def forward(self, x, encoder_hidden_states=None):
        # the input is [B, C, H, W]
        B, T, C = x.shape
        # print(f"Attn in shape { x.shape}")
        q = self.to_q(x)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else x
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # q [B, H*W, C], k [B, T, C],
        q = q.view(B, -1, self.n_head, C//self.n_head).transpose(1, 2)
        k = k.view(B, -1, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, -1, self.n_head, C//self.n_head).transpose(1, 2)
        # print(f"Q shape {q.shape} and k shape {k.shape}")
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # attn [B, H*W, NH, NDIM]
        x = x.transpose(1, 2).contiguous().view(B, -1, C)

        x = self.to_out(x)

        return x

class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate)

    def forward(self, hidden_states, ):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)

class MLP(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.input_channels = input_channels
        self.geglu = GEGLU(input_channels, 4*input_channels, bias=True)
        self.proj_out = nn.Linear(input_channels*4, input_channels, bias=True)

    def forward(self, x):
        x = self.geglu(x)
        x = self.proj_out(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(self, input_channels, cross_attention_dim, n_head, norm="LN",
                 qv_norm=None):
        super().__init__()
        self.input_channels = input_channels
        self.cross_attention_dim = cross_attention_dim
        if norm == "LN":
            self.norm1 = nn.LayerNorm((self.input_channels,))
            self.norm2 = nn.LayerNorm((self.input_channels,))
            self.norm3 = nn.LayerNorm((self.input_channels,))
        elif norm =="RSM":
            self.norm1 = nn.RMSNorm(self.input_channels, eps=1e-6)
            self.norm2 = nn.RMSNorm(self.input_channels, eps=1e-6)
            self.norm3 = nn.RMSNorm(self.input_channels, eps=1e-6)
        else:
            raise ValueError

        self.attn1 = Attention(self.input_channels, n_head, qv_norm=qv_norm)
        self.attn2 = Attention(self.input_channels, n_head,
                               self.cross_attention_dim, qv_norm=qv_norm)

        self.ff = MLP(self.input_channels)

    def forward(self, x, encoder_hidden_states):
        hidden_states = x + self.attn1(self.norm1(x))
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states)
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))

        return hidden_states


class AttentionBlock(nn.Module):

    def __init__(self, input_channels, cross_attention_dim, n_head, num_groups,
                 eps, norm="LN", qv_norm=None):
        super().__init__()

        assert input_channels % n_head == 0
        self.input_channels = input_channels
        self.cross_attention_dim = cross_attention_dim
        self.num_groups = num_groups
        self.eps = eps

        self.norm = nn.GroupNorm(num_groups=self.num_groups,
                            num_channels=self.input_channels,
                            eps=1e-6)
        self.proj_in = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels,
                                kernel_size=1, stride=1, padding=0, bias=True)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(input_channels,
                                              cross_attention_dim, n_head,
                                              norm=norm, qv_norm=qv_norm)])

        self.proj_out = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels,
                                kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, encoder_hidden_states):
        batch, _, height, width = x.shape
        res = x
        # prepare continuous input for transformers
        x = self.norm(x)
        x = self.proj_in(x)
        inner_dim = x.shape[1]
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states)

        # prepare output
        x = x.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        x = self.proj_out(x)
        x = x + res

        return x

@dataclass
class UnetConfig:
    in_channels: int = 3
    out_channels: int = 3
    block_out_channels: ClassVar[list[int]] = [128, 256, 256, 256]
    cross_attention_dim: int = 512
    num_blocks: int = 4
    attention_head_dim: int = 8
    layers_per_block: int = 2
    norm_num_groups: int = 32
    norm_eps: int = 1e-05
    num_classes: int = 10
    norm: str="LN"
    qv_norm: str=None

class Unet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        block_out_channels = config.block_out_channels
        time_embed_dim = block_out_channels[-1] # Usually time dim matches max channels
        self.in_channels = config.in_channels
        self.class_embedding = nn.Embedding(
               config.num_classes, config.cross_attention_dim # e.g., 10 classes, 64 dim
           )

        # 1. Input Convolution
        self.conv_in = nn.Conv2d(
            config.in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1
        )

        # 2. Time Embedding
        self.time_embedding = TimeEmbeddings(
            sinusoidal_dim=block_out_channels[0],
            output_dim=time_embed_dim,
        )

        # down_blocks is Resnet  ->  CrossAtten -> Downsample except last block is just Resnet
        # is the AttentionBlock the one with the double the channels for the up block in the unet
        # and is the resnet the one how changes the block channels
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i in range(config.num_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == config.num_blocks - 1

            # Create a regular ModuleList for resnets and attentions instead of ModuleDict
            resnets = nn.ModuleList([])
            attentions = nn.ModuleList([])

            for j in range(config.layers_per_block):
                # First resnet in block handles channel changes
                res_input_channel = input_channel if j == 0 else output_channel
                resnets.append(
                    ResnetBlock(
                        res_input_channel,
                        output_channel,
                        time_embed_dim,
                        config.norm_num_groups,
                        config.norm_eps
                    )
                )
                # Add attention blocks except for the final block
                if not is_final_block:
                    attentions.append(
                        AttentionBlock(
                            output_channel,
                            config.cross_attention_dim,
                            config.attention_head_dim,
                            config.norm_num_groups,
                            config.norm_eps,
                            config.norm,
                            config.qv_norm
                        )
                    )

            # Create a module dictionary for the entire block
            down_block = nn.ModuleDict({
                "resnets": resnets,
                "attentions": attentions,
            })

            # Add downsamplers at the block level, not nested inside a ModuleList
            if not is_final_block:
                # Fixed: Use a direct Conv2d module, not wrapped in another ModuleList
                down_block["downsamplers"] = nn.ModuleList([
                    nn.Conv2d(
                        output_channel, output_channel,
                        kernel_size=3, stride=2, padding=1
                    )
                ])

            self.down_blocks.append(down_block)

        self.mid_block = nn.ModuleDict({
            "resnets": nn.ModuleList([
                ResnetBlock(
                    block_out_channels[-1],
                    block_out_channels[-1],
                    time_embed_dim,
                    config.norm_num_groups,
                    config.norm_eps
                ),
                ResnetBlock(
                    block_out_channels[-1],
                    block_out_channels[-1],
                    time_embed_dim,
                    config.norm_num_groups,
                    config.norm_eps
                ),
            ]),
            "attentions": nn.ModuleList([
                AttentionBlock(
                    block_out_channels[-1],
                    config.cross_attention_dim,
                    config.attention_head_dim,
                    config.norm_num_groups,
                    config.norm_eps,
                    config.norm,
                    config.qv_norm
                )
            ])
        })

        # 5. Up Blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        num_layers = config.layers_per_block + 1
        for i in range(config.num_blocks):
            prev_output_channel = output_channel#reversed_block_out_channels[max(i - 1, 0)]
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(reversed_block_out_channels) - 1)]

            is_first_block = i == 0

            # Create ModuleLists for resnets and attentions
            resnets = nn.ModuleList([])
            attentions = nn.ModuleList([])
            # Fixed: Calculate skip connection channels correctly for each layer
            for j in range(num_layers):
                res_skip_channels = input_channel if (j == num_layers - 1) else output_channel
                resnet_in_channels = prev_output_channel if j == 0 else output_channel
                resnets.append(
                    ResnetBlock(
                        resnet_in_channels + res_skip_channels,
                        output_channel,
                        time_embed_dim,
                        config.norm_num_groups,
                        config.norm_eps
                    )
                )

                # Add attention blocks (in up blocks they come after resnets)
                if not is_first_block:
                    attentions.append(
                        AttentionBlock(
                            output_channel,
                            config.cross_attention_dim,
                            config.attention_head_dim,
                            config.norm_num_groups,
                            config.norm_eps,
                            config.norm,
                            config.qv_norm
                        )
                    )

            # Create module dictionary for the entire block
            up_block = nn.ModuleDict({
                "resnets": resnets,
                "attentions": attentions
            })

            # Add upsamplers at the block level
            if i < config.num_blocks - 1:  # No upsampler needed for the last block
                up_block["upsamplers"] = nn.ModuleList([
                    UpsamplerBlock(output_channel, output_channel)
                ])

            self.up_blocks.append(up_block)

        # 6. Output Convolution
        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups,
            num_channels=block_out_channels[0], # Final output channels match first block
            eps=config.norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            config.out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x, timestep, encoder_hidden_states):
        # class
        encoder_hidden_states = self.class_embedding(encoder_hidden_states)
        # 1. time
        # t_emb = self.time_proj(timestep)
        t_emb = self.time_embedding(timestep, x)
        # 2. preprocess
        x = self.conv_in(x)

        # 3. down
        down_block_res_x = (x,)

        for i, downsample_block in enumerate(self.down_blocks):
            output_states = ()
            if i != (self.config.num_blocks-1):
                for resnet, attention in zip(downsample_block.resnets, downsample_block.attentions):
                    x = resnet(x, t_emb)
                    x = attention(x, encoder_hidden_states)
                    output_states += (x, )

                for downsamplers in downsample_block.downsamplers:
                    x = downsamplers(x)
                output_states = output_states + (x,)

            else:
                for resnet in downsample_block.resnets:
                    x =resnet(x, t_emb)
                    output_states += (x, )

            down_block_res_x += output_states

        # 4. mid
        x = self.mid_block.resnets[0](x, t_emb)
        for attention, resnet,  in zip(self.mid_block.attentions, self.mid_block.resnets[1:]):
            x = attention(x, encoder_hidden_states)
            x = resnet(x, t_emb)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            # gets the last 3 outputs of the downblocks (the resnets)
            res_x_tuple = down_block_res_x[-len(upsample_block.resnets):]
            down_block_res_x = down_block_res_x[:-len(upsample_block.resnets)]
            if i == 0:
                for resnet in upsample_block.resnets:
                    res_x = res_x_tuple[-1]
                    res_x_tuple = res_x_tuple[:-1]
                    x = torch.cat([x, res_x], dim=1)

                    x = resnet(x, t_emb)

                for upsampler in upsample_block.upsamplers:
                    x = upsampler(x)

            elif i == len(self.up_blocks)-1:
                for resnet, attention in zip(upsample_block.resnets, upsample_block.attentions):
                    res_x = res_x_tuple[-1]
                    res_x_tuple = res_x_tuple[:-1]
                    x = torch.cat([x, res_x], dim=1)

                    x = resnet(x, t_emb)
                    x = attention(x, encoder_hidden_states)

            else:
                for resnet, attention in zip(upsample_block.resnets, upsample_block.attentions):
                    res_x = res_x_tuple[-1]
                    res_x_tuple = res_x_tuple[:-1]
                    x = torch.cat([x, res_x], dim=1)

                    x = resnet(x, t_emb)
                    x = attention(x, encoder_hidden_states)

                for upsampler in upsample_block.upsamplers:
                    x = upsampler(x)

        # 6. post-process
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x

def weights_init(m, activation_type='relu', leaky_relu_slope=0.01):
    """
    Applies Kaiming initialization to convolutional and linear layers.
    Initializes normalization layers appropriately.

    Args:
        m (nn.Module): The module to initialize.
        activation_type (str): The type of nonlinearity used after conv/linear
                               layers. Used by Kaiming init.
                               Defaults to 'relu'. For SiLU/Swish or GELU
                               activations, using 'relu' is a common and
                               effective practice.
        leaky_relu_slope (float): The negative slope for leaky_relu,
                                  if 'leaky_relu' is chosen as
                                  activation_type. Defaults to 0.01.
    """
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
            m.weight.data,
            # 'a' is the negative slope for leaky_relu.
            # It's ignored if nonlinearity is 'relu'.
            a=leaky_relu_slope if activation_type == 'leaky_relu' else 0,
            mode='fan_in',
            nonlinearity=activation_type
        )
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(
            m.weight.data,
            a=leaky_relu_slope if activation_type == 'leaky_relu' else 0,
            mode='fan_in',
            nonlinearity=activation_type
        )
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
