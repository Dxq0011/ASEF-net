"""ASEF-Net: latent exposure ODE, PIEMF and confidence-aware reconstruction.

This implementation follows the equations and shape contracts described in the
ASEF-Net V3.6 manuscript. Architecture details not fixed by the manuscript
(e.g. residual-block counts) are explicit parameters below.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
        )
    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x + self.net(x))


def _groups(c: int) -> int:
    return max(g for g in (1, 2, 4, 8, 16, 32) if c % g == 0)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int, blocks: int = 1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1),
                                  nn.GroupNorm(_groups(cout), cout), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualBlock(cout) for _ in range(blocks)))
    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.conv(x))


class Up(nn.Module):
    """Upsampling path of the shared decoder; no skip connections (Eq. 15)."""
    def __init__(self, cin: int, cout: int, blocks: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(_groups(cout), cout), nn.SiLU(),
            *(ResidualBlock(cout) for _ in range(blocks)),
        )
    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.net(x)


class Encoder(nn.Module):
    """Produces only the bottleneck feature (Sec. 3.2)."""
    def __init__(self, base: int = 64, channels: int = 512):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, base, 3, padding=1),
                                  nn.GroupNorm(_groups(base), base), nn.SiLU(),
                                  ResidualBlock(base))
        self.d1 = Down(base, base*2)
        self.d2 = Down(base*2, base*4, blocks=2)
        self.d3 = Down(base*4, channels, blocks=2)
    def forward(self, x: Tensor) -> Tensor:
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        return self.d3(s2)


class Decoder(nn.Module):
    """Shared decoder D(Z_t): reconstructs each stage image from the latent
    state alone (Eq. 15); all exposure stages share these parameters."""
    def __init__(self, base: int = 64, channels: int = 512):
        super().__init__()
        self.up2 = Up(channels, base*4, blocks=2)
        self.up1 = Up(base*4, base*2)
        self.up0 = Up(base*2, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1)
    def forward(self, z: Tensor) -> Tensor:
        x = self.up2(z)
        x = self.up1(x)
        x = self.up0(x)
        return torch.sigmoid(self.out(x))


class FieldBranch(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1),
                                  nn.SiLU(), nn.Conv2d(hidden, 1, 1))
    def forward(self, z: Tensor):
        return self.body(z)


class PositionEncoding(nn.Module):
    """Continuous exposure embedding, evaluated also at RK4 midpoint times."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, channels), nn.SiLU(),
                                 nn.Linear(channels, channels))
    def forward(self, e: float, z: Tensor):
        time = z.new_full((z.shape[0], 1), float(e))
        return self.net(time).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, z.shape[-2], z.shape[-1])


class Dynamics(nn.Module):
    """f_theta([Z,n,c,p(e)]), returns a latent-state derivative feature map.

    Convolutional layers with GroupNorm normalization and SiLU activation,
    matching the manuscript's dynamics-function description.
    """
    def __init__(self, channels: int, emb: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 2 + emb, hidden, 1),
            nn.GroupNorm(_groups(hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GroupNorm(_groups(hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        # Small initial dynamics improve stability without imposing hard clipping.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, z: Tensor, n: Tensor, c: Tensor, p: Tensor) -> Tensor:
        return self.net(torch.cat([z, n, c, p], dim=1))


class PIEMF(nn.Module):
    """R = softplus(R_g(global_pool(Z)) + eps * tanh(Delta_R(Z)))."""
    def __init__(self, channels: int, hidden: int, epsilon: float = .1):
        super().__init__()
        self.epsilon = epsilon
        self.global_response = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, 1, 1))
        self.local_response = nn.Sequential(nn.Conv2d(channels, hidden, 1),
            nn.SiLU(), nn.Conv2d(hidden, 1, 3, padding=1))
        # Initialize positive response near ~1 while preserving learnability.
        nn.init.constant_(self.global_response[-1].bias, 1.0)
    def forward(self, z: Tensor, local_enabled: bool = True):
        shared = self.global_response(z)
        deviation = self.epsilon * torch.tanh(self.local_response(z)) if local_enabled else torch.zeros_like(z[:, :1])
        return F.softplus(shared + deviation)


@dataclass
class ASEFConfig:
    channels: int = 512
    base: int = 64
    field_hidden: int = 96
    dynamics_hidden: int = 176
    exposure_emb: int = 32
    attention_hidden: int = 192
    stages: int = 10
    rk4_steps: int = 10
    epsilon: float = 0.1
    eta: float = 1e-6
    variant: str = 'full'
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.stages < 1 or self.rk4_steps < 1:
            raise ValueError('stages and rk4_steps must be positive')
        if self.channels != 8*self.base:
            raise ValueError('this encoder requires channels == base*8')
        if self.variant not in ('full','v-spatial-mean','n-spatial-mean','c-spatial-mean',
                                'no-local-PIEMF','mean-fusion','last-stage-only'):
            raise ValueError(f'Unknown variant: {self.variant}')


class ASEFNet(nn.Module):
    """Inputs and outputs are normalized RGB in [0,1].

    count_rhs_calls includes the four stages of every standard RK4 step,
    rather than omitting encoder-associated or boundary work.
    """
    def __init__(self, cfg: ASEFConfig | None = None):
        super().__init__()
        self.cfg = cfg or ASEFConfig()
        c = self.cfg
        self.encoder = Encoder(c.base, c.channels)
        self.decoder = Decoder(c.base, c.channels)
        self.vbranch = FieldBranch(c.channels, c.field_hidden)
        self.nbranch = FieldBranch(c.channels, c.field_hidden)
        self.cbranch = FieldBranch(c.channels, c.field_hidden)
        self.position = PositionEncoding(c.exposure_emb)
        self.dynamics = Dynamics(c.channels, c.exposure_emb, c.dynamics_hidden)
        self.piemf = PIEMF(c.channels, c.field_hidden, c.epsilon)
        self.attention = nn.Sequential(
            nn.Conv2d(3*c.stages, c.attention_hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c.attention_hidden, c.attention_hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c.attention_hidden, c.stages, 3, padding=1),
        )
        self.count_rhs_calls = 0

    def _rhs(self, e: float, z: Tensor, v: Tensor, n: Tensor, c: Tensor) -> Tensor:
        self.count_rhs_calls += 1
        p = self.position(e, z)
        f = self.dynamics(z, n, c, p)
        r = self.piemf(z, local_enabled=(self.cfg.variant!='no-local-PIEMF'))
        return (1.0 + v) * f / (r + self.cfg.eta)

    def _call_rhs(self, e: float, z: Tensor, v: Tensor, n: Tensor, c: Tensor):
        if self.cfg.gradient_checkpointing and self.training and torch.is_grad_enabled():
            # non-reentrant checkpointing saves memory across RK4 steps.
            return checkpoint(lambda zz, ee=e: self._rhs(ee, zz, v, n, c), z, use_reentrant=False)
        return self._rhs(e,z,v,n,c)

    def forward(self, image: Tensor) -> Dict[str,Tensor]:
        if image.ndim != 4 or image.shape[1]!=3:
            raise ValueError('Expected RGB image tensor [B,3,H,W]')
        if image.shape[-2] % 8 or image.shape[-1] % 8:
            raise ValueError('Input H and W must be multiples of 8')
        z = self.encoder(image)
        v_raw = F.softplus(self.vbranch(z))
        n_raw = torch.sigmoid(self.nbranch(z))
        c_raw = torch.sigmoid(self.cbranch(z))
        v,n,c=v_raw,n_raw,c_raw
        if self.cfg.variant == 'v-spatial-mean':
            v = v.mean((2,3),keepdim=True).expand_as(v)
        elif self.cfg.variant == 'n-spatial-mean':
            n = n.mean((2,3),keepdim=True).expand_as(n)
        elif self.cfg.variant == 'c-spatial-mean':
            c = c.mean((2,3),keepdim=True).expand_as(c)
        stages, steps = self.cfg.stages, self.cfg.rk4_steps
        h = 1.0/(stages*steps)
        e = 0.0
        self.count_rhs_calls = 0
        decoded=[]
        for t in range(stages):
            for _ in range(steps):
                k1=self._call_rhs(e, z, v,n,c)
                k2=self._call_rhs(e+h/2,z+h*k1/2,v,n,c)
                k3=self._call_rhs(e+h/2,z+h*k2/2,v,n,c)
                k4=self._call_rhs(e+h,z+h*k3,v,n,c)
                z=z+(h/6)*(k1+2*k2+2*k3+k4)
                e+=h
            if self.cfg.variant != 'last-stage-only' or t == stages - 1:
                decoded.append(self.decoder(z))
        stage_images=torch.stack(decoded,dim=1)
        if self.cfg.variant == 'last-stage-only':
            weights=torch.ones_like(stage_images[:,:,:1])
        elif self.cfg.variant == 'mean-fusion':
            weights=torch.full_like(stage_images[:,:,:1],1.0/stages)
        else:
            logits=self.attention(torch.cat(decoded,dim=1))
            weights=torch.softmax(logits,dim=1).unsqueeze(2)
        output=(stage_images*weights).sum(dim=1)
        return {'output':output,'stages':stage_images,'weights':weights,
                'v':v, 'n':n,'c':c,'v_raw':v_raw,'n_raw':n_raw,'c_raw':c_raw}

    def parameters_millions(self):
        return sum(p.numel() for p in self.parameters())/1e6
