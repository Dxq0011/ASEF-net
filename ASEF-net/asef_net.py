"""
ASEF-Net: Asynchronous Spatial Exposure Fields for Low-Light Image Enhancement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import NAFBlock, EdgeAwareSmooth, ODEMLP


class SharedEncoder(nn.Module):
    """Shared feature encoder: extracts multi-scale features"""
    def __init__(self, in_ch=3, base_ch=32, num_blocks=[2, 3, 4]):
        super().__init__()

        self.intro = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        # Encoder stages
        self.enc1 = nn.Sequential(*[NAFBlock(base_ch) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 2, 2)  # /2

        self.enc2 = nn.Sequential(*[NAFBlock(base_ch * 2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 2, 2)  # /4

        self.enc3 = nn.Sequential(*[NAFBlock(base_ch * 4) for _ in range(num_blocks[2])])

        # Decoder (lightweight)
        self.up2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.dec2 = nn.Sequential(*[NAFBlock(base_ch * 2) for _ in range(2)])

        self.up1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.dec1 = nn.Sequential(*[NAFBlock(base_ch) for _ in range(2)])

        self.out_conv = nn.Conv2d(base_ch, base_ch, 3, 1, 1)

    def forward(self, x):
        """Returns feature map at original resolution"""
        x1 = self.intro(x)

        x1 = self.enc1(x1)
        x2 = self.down1(x1)

        x2 = self.enc2(x2)
        x3 = self.down2(x2)

        x3 = self.enc3(x3)

        x2_up = self.up2(x3)
        x2 = self.dec2(x2_up + x2)

        x1_up = self.up1(x2)
        x1 = self.dec1(x1_up + x1)

        feat = self.out_conv(x1)
        return feat


class FieldHeads(nn.Module):
    """Predict v(x,y), n(x,y), c(x,y) from shared features"""
    def __init__(self, feat_ch=32):
        super().__init__()

        # v(x,y): exposure velocity, must be >= 0
        self.head_v = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, 1, 1)
        )
        self.v_bias = 1.0

        # n(x,y): noise confidence, in [0, 1]
        self.head_n = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, 1, 1),
            nn.Sigmoid()
        )

        # c(x,y): color stability, in [0, 1]
        self.head_c = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        """
        Returns:
            v: [B, 1, H, W], >= 0
            n: [B, 1, H, W], in [0,1]
            c: [B, 1, H, W], in [0,1]
        """
        v = F.softplus(self.head_v(feat)) + self.v_bias  # >= v_bias, init ~1.0
        n = self.head_n(feat)
        c = self.head_c(feat)
        return v, n, c


class ODESolver(nn.Module):
    """ODE solver for spatial exposure evolution"""
    def __init__(self, ch=3, feat_ch=32, num_steps=10, lambda_coupling=0.01, use_spatial_coupling=True):
        super().__init__()
        self.num_steps = num_steps
        self.lambda_coupling = lambda_coupling
        self.use_spatial_coupling = use_spatial_coupling

        # ODE function f_theta
        self.ode_fn = ODEMLP(ch, hidden_dim=ch * 4)

        # Condition fusion: combine v, n, c into condition feature
        self.cond_fusion = nn.Conv2d(3, ch, 1)

        # Spatial coupling
        if use_spatial_coupling:
            self.spatial_coupling = EdgeAwareSmooth(gamma=5.0)

    def forward(self, z0, v, n, c, I_low):
        """
        Solve ODE from e=0 to e=E_max using Euler method
        z0: [B, 3, H, W] - initial state (low-light input)
        v, n, c: [B, 1, H, W] - spatial fields
        Returns: list of states [z_0, z_1, ..., z_K]
        """
        B, C, H, W = z0.shape

        # Prepare condition feature
        cond_input = torch.cat([v, n, c], dim=1)  # [B, 3, H, W]
        cond = self.cond_fusion(cond_input)  # [B, C, H, W]

        # Fixed base step
        delta_e = 1.0 / self.num_steps

        states = [z0]
        z = z0

        for k in range(self.num_steps):
            # ODE derivative scaled by v (exposure velocity)
            # v controls the actual magnitude of enhancement
            # Ensure dz >= 0 so the ODE always brightens (not darkens)
            dz = v * F.softplus(self.ode_fn(z, cond))

            # Apply noise-aware damping: n close to 1 -> dampen derivative (conservative)
            noise_damp = 1.0 - 0.5 * n  # [B, 1, H, W]
            dz = dz * noise_damp

            # Spatial coupling (edge-aware smoothness)
            if self.use_spatial_coupling:
                coupling = self.spatial_coupling(z, I_low)
                dz = dz + self.lambda_coupling * coupling

            # Euler step with fixed step size
            z = z + delta_e * dz

            # Keep channel evolution independent here. Pulling RGB channels toward their
            # mean made enhanced images visibly desaturated on LOL-style samples.

            states.append(z)

        # Only clamp at the very end
        z = torch.clamp(z, 0, 1)
        states[-1] = z

        return states


class FusionModule(nn.Module):
    """Credibility-weighted fusion of multiple exposure stages"""
    def __init__(self, ch=3, num_stages=6):
        super().__init__()
        self.num_stages = num_stages

        # For each stage, predict a weight map from (stage_state - input)
        self.weight_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2, ch, 3, 1, 1),  # cat(stage, input)
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch // 2, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 2, 1, 1)
            ) for _ in range(num_stages)
        ])

    def forward(self, states, I_low):
        """
        states: list of [B, 3, H, W], length = num_stages
        I_low: [B, 3, H, W]
        Returns: [B, 3, H, W]
        """
        B, C, H, W = I_low.shape

        weights = []
        for i, state in enumerate(states):
            diff = state - I_low
            w_input = torch.cat([state, diff], dim=1)
            w = self.weight_nets[i](w_input)  # [B, 1, H, W]
            weights.append(w)

        # Stack and softmax over stages
        weights = torch.stack(weights, dim=1)  # [B, K, 1, H, W]
        weights = F.softmax(weights, dim=1)

        # Stack states
        states_tensor = torch.stack(states, dim=1)  # [B, K, 3, H, W]

        # Weighted sum
        weights = weights.expand(-1, -1, C, -1, -1)  # [B, K, 3, H, W]
        I_out = (weights * states_tensor).sum(dim=1)  # [B, 3, H, W]

        return torch.clamp(I_out, 0, 1)


class RefinementModule(nn.Module):
    """Learnable color/detail correction after exposure fusion."""
    def __init__(self, in_ch=3, feat_ch=32):
        super().__init__()
        self.intro = nn.Conv2d(in_ch * 2 + feat_ch, feat_ch, 3, 1, 1)
        self.body = nn.Sequential(
            NAFBlock(feat_ch),
            NAFBlock(feat_ch),
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, in_ch, 3, 1, 1)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.2))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, I_low, I_fused, feat):
        x = torch.cat([I_low, I_fused, feat], dim=1)
        residual = torch.tanh(self.body(self.intro(x))) * self.res_scale.clamp(0.0, 0.5)
        return torch.clamp(I_fused + residual, 0, 1)


class ASEFNet(nn.Module):
    """Complete ASEF-Net"""
    def __init__(self, 
                 in_ch=3, 
                 base_ch=32, 
                 num_blocks=[2, 3, 4],
                 ode_steps=5,
                 lambda_coupling=0.01,
                 use_spatial_coupling=True):
        super().__init__()

        self.encoder = SharedEncoder(in_ch, base_ch, num_blocks)
        self.field_heads = FieldHeads(base_ch)
        self.ode_solver = ODESolver(in_ch, base_ch, ode_steps, lambda_coupling, use_spatial_coupling)
        self.fusion = FusionModule(in_ch, num_stages=ode_steps + 1)
        self.refiner = RefinementModule(in_ch, base_ch)

        self.ode_steps = ode_steps

    def forward(self, I_low):
        """
        I_low: [B, 3, H, W] in [0, 1]
        Returns:
            I_out: [B, 3, H, W] enhanced image
            v: [B, 1, H, W] exposure velocity field
            n: [B, 1, H, W] noise confidence field
            c: [B, 1, H, W] color stability field
            states: list of intermediate exposure stages
        """
        # Extract shared features
        feat = self.encoder(I_low)

        # Predict three spatial fields
        v, n, c = self.field_heads(feat)

        # Solve ODE to generate virtual exposure stages
        states = self.ode_solver(I_low, v, n, c, I_low)

        # Fuse stages
        I_fused = self.fusion(states, I_low)
        I_out = self.refiner(I_low, I_fused, feat)

        return I_out, v, n, c, states


class ASEFNetLite(nn.Module):
    """Lightweight version: fewer blocks, fewer ODE steps, no spatial coupling"""
    def __init__(self, in_ch=3, base_ch=16, ode_steps=10):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, base_ch, num_blocks=[1, 2, 2])
        self.field_heads = FieldHeads(base_ch)
        self.ode_solver = ODESolver(in_ch, base_ch, ode_steps, 
                                     lambda_coupling=0.0, use_spatial_coupling=False)
        self.fusion = FusionModule(in_ch, num_stages=ode_steps + 1)
        self.refiner = RefinementModule(in_ch, base_ch)
        self.ode_steps = ode_steps

    def forward(self, I_low):
        feat = self.encoder(I_low)
        v, n, c = self.field_heads(feat)
        states = self.ode_solver(I_low, v, n, c, I_low)
        I_fused = self.fusion(states, I_low)
        I_out = self.refiner(I_low, I_fused, feat)
        return I_out, v, n, c, states
