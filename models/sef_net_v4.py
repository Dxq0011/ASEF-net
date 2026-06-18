"""
SEF-Net: Learning Spatial Exposure Flows for Low-Light Image Enhancement
A Physics-Informed Exposure State Field Framework

Five-Module Architecture (Strictly Aligned with Design):
Module 1: Exposure Initialization Encoder (Pure Encoder, 512xH/8xW/8)
Module 2: Exposure State Field Constructor (Feature Space, Tx512xH/8xW/8)
Module 3: Condition Field Generation (Three Branches at H/8xW/8)
Module 4: Physics-Informed Spatial Exposure Dynamics (Neural ODE in Feature Space)
Module 5: Exposure Trajectory Reconstruction (Shared Decoder + Adaptive Fusion)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import CharbonnierLoss, PerceptualLoss, ColorLoss, SaturationLoss, ContrastLoss, ToneLoss


class ResBlock(nn.Module):
    """Residual Block: Conv+BN+ReLU with skip connection."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 1, padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# =============================================================================
# Module 1: Exposure Initialization Encoder (Pure Encoder)
# =============================================================================
class ExposureInitEncoder(nn.Module):
    """
    Pure Encoder (U-Net style, downsampling path only).
    Input: I_0 (3xHxW)
    Output: F_0 (512xH/8xW/8) - bottleneck feature
    """
    def __init__(self, in_ch=3):
        super().__init__()
        
        # Stage 1: 64 channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            ResBlock(64)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: 128 channels
        self.stage2 = nn.Sequential(
            ResBlock(128),
            ResBlock(128)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: 256 channels
        self.stage3 = nn.Sequential(
            ResBlock(256),
            ResBlock(256)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck: 512 channels
        self.bottleneck = nn.Sequential(
            ResBlock(512),
            ResBlock(512)
        )
    
    def forward(self, x):
        x1 = self.stage1(x)           # [B, 64, H, W]
        x2 = self.down1(x1)           # [B, 128, H/2, W/2]
        
        x2 = self.stage2(x2)          # [B, 128, H/2, W/2]
        x3 = self.down2(x2)           # [B, 256, H/4, W/4]
        
        x3 = self.stage3(x3)          # [B, 256, H/4, W/4]
        x4 = self.down3(x3)           # [B, 512, H/8, W/8]
        
        f0 = self.bottleneck(x4)      # [B, 512, H/8, W/8]
        skips = [x3, x2, x1]          # deep -> shallow for decoder
        return f0, skips


# =============================================================================
# Module 2: Exposure State Field Constructor
# =============================================================================
class ExposureStateFieldConstructor(nn.Module):
    """
    Construct initial Exposure State Field in feature space.
    Input: F_0 (512xH/8xW/8)
    Output: Z_0 (Tx512xH/8xW/8)
    """
    def __init__(self, T=10, C_emb=64, feat_ch=512):
        super().__init__()
        self.T = T
        self.feat_ch = feat_ch
        
        # Learnable exposure embeddings
        self.exposure_emb = nn.Parameter(torch.randn(T, C_emb))
        
        # Projection to feature space
        self.proj = nn.Linear(C_emb, feat_ch)
    
    def forward(self, f0):
        """
        f0: [B, 512, H/8, W/8]
        Returns: [B, T, 512, H/8, W/8]
        """
        B, C, H, W = f0.shape
        
        # Project embeddings
        projected = self.proj(self.exposure_emb)  # [T, 512]
        projected = projected.view(self.T, self.feat_ch, 1, 1)  # [T, 512, 1, 1]
        
        # Broadcast and add to f0
        f0_expanded = f0.unsqueeze(1)  # [B, 1, 512, H, W]
        projected_expanded = projected.unsqueeze(0)  # [1, T, 512, 1, 1]
        
        Z0 = f0_expanded + projected_expanded  # [B, T, 512, H, W]
        return Z0


# =============================================================================
# Module 3: Condition Field Generation (Three Branches)
# =============================================================================
class ConditionFieldGeneration(nn.Module):
    """
    Three-branch field prediction from F_0.
    Input: F_0 (512xH/8xW/8)
    Output: v, n, c (each 1xH/8xW/8)
    """
    def __init__(self, feat_ch=512):
        super().__init__()
        
        # Velocity branch
        self.velocity_branch = nn.Sequential(
            nn.Conv2d(feat_ch, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Softplus()  # v >= 0
        )
        
        # Noise branch
        self.noise_branch = nn.Sequential(
            nn.Conv2d(feat_ch, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # n in [0, 1]
        )
        
        # Color branch
        self.color_branch = nn.Sequential(
            nn.Conv2d(feat_ch, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # c in [0, 1]
        )
    
    def forward(self, f0):
        v = self.velocity_branch(f0)
        n = self.noise_branch(f0)
        c = self.color_branch(f0)
        return v, n, c


# =============================================================================
# Module 4: Physics-Informed Spatial Exposure Dynamics (Neural ODE)
# =============================================================================
class EERF(nn.Module):
    """
    Effective Exposure Response Field (EERF).
    R(x,y,z) = R_g(z) + ε * tanh(ΔR_φ(x,y,z))
    Constraint hardcoded via tanh: |ΔR| < ε
    """
    def __init__(self, feat_ch=512, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        
        # Global response (shared weights)
        self.R_global = nn.Conv2d(feat_ch, feat_ch, 1)
        
        # Spatial deviation (pixel-independent, bounded by tanh)
        self.R_delta = nn.Conv2d(feat_ch, feat_ch, 1)
    
    def forward(self, z):
        """
        z: [B, 512, H, W]
        Returns: [B, 512, H, W]
        """
        R_g = self.R_global(z)
        delta_R = self.R_delta(z)
        R = R_g + self.epsilon * torch.tanh(delta_R)
        return R


class NeuralODEFunc(nn.Module):
    """
    Neural ODE function f_θ.
    Input: [Z_t, v, n, c] concatenated
    Output: ∂Z/∂e
    """
    def __init__(self, feat_ch=512):
        super().__init__()
        
        self.ode_func = nn.Sequential(
            nn.Conv2d(feat_ch + 3, feat_ch, 3, padding=1),
            nn.GroupNorm(32, feat_ch),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1)
        )
        
        # Initialize last layer to zero for stability
        nn.init.zeros_(self.ode_func[-1].weight)
        nn.init.zeros_(self.ode_func[-1].bias)
    
    def forward(self, z, v, n, c):
        """
        z: [B, 512, H, W]
        v, n, c: [B, 1, H, W]
        Returns: [B, 512, H, W]
        """
        # Broadcast v, n, c to match z resolution if needed
        if v.shape[2:] != z.shape[2:]:
            v = F.interpolate(v, size=z.shape[2:], mode='bilinear', align_corners=False)
            n = F.interpolate(n, size=z.shape[2:], mode='bilinear', align_corners=False)
            c = F.interpolate(c, size=z.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([z, v, n, c], dim=1)
        dz_de = self.ode_func(x)
        return dz_de


class SpatialExposureFlowSolver(nn.Module):
    """
    RK4 Solver for Spatial Exposure Flow ODE.
    ∂z/∂e = f_θ([z,v,n,c]) ⊙ R(x,y,z)^{-1}
    """
    def __init__(self, ode_func, eerf, num_steps=10):
        super().__init__()
        self.ode_func = ode_func
        self.eerf = eerf
        self.num_steps = num_steps
    
    def forward(self, Z0, v, n, c):
        """
        Z0: [B, 512, H, W] - initial exposure state
        v, n, c: [B, 1, H, W] - condition fields
        Returns: list of [B, 512, H, W] - exposure trajectory
        """
        B, C, H, W = Z0.shape
        device = Z0.device
        
        delta_e = 1.0 / self.num_steps
        
        states = [Z0]
        z = Z0
        
        for k in range(self.num_steps):
            # RK4 stages with gradient clipping
            R_z = self.eerf(z)
            dz1 = self.ode_func(z, v, n, c)
            k1 = dz1 / (R_z.abs() + 0.1)  # Avoid division by small values
            k1 = torch.clamp(k1, -10, 10)
            
            z2 = z + 0.5 * delta_e * k1
            R_z2 = self.eerf(z2)
            dz2 = self.ode_func(z2, v, n, c)
            k2 = dz2 / (R_z2.abs() + 0.1)
            k2 = torch.clamp(k2, -10, 10)
            
            z3 = z + 0.5 * delta_e * k2
            R_z3 = self.eerf(z3)
            dz3 = self.ode_func(z3, v, n, c)
            k3 = dz3 / (R_z3.abs() + 0.1)
            k3 = torch.clamp(k3, -10, 10)
            
            z4 = z + delta_e * k3
            R_z4 = self.eerf(z4)
            dz4 = self.ode_func(z4, v, n, c)
            k4 = dz4 / (R_z4.abs() + 0.1)
            k4 = torch.clamp(k4, -10, 10)
            
            z = z + (delta_e / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            z = torch.clamp(z, -5, 5)  # Prevent explosion
            states.append(z)
        
        return states


# =============================================================================
# Module 5: Exposure Trajectory Reconstruction
# =============================================================================
class SharedDecoderHead(nn.Module):
    """
    Shared Decoder Head D with U-Net skip connections.
    Input: Z_et (512xH/8xW/8), skip features from encoder
    Output: I_t (3xHxW)
    All T stages share the same decoder weights.
    """
    def __init__(self, feat_ch=512, skip_chs=[256, 128, 64]):
        super().__init__()
        
        # Stage 1: H/8 -> H/4
        self.up1_conv = nn.Conv2d(feat_ch, 256, 3, padding=1)
        self.up1_ps = nn.PixelShuffle(2)  # 256 -> 64 channels
        self.up1_norm = nn.InstanceNorm2d(64)
        self.up1_relu = nn.ReLU(inplace=True)
        self.skip1_proj = nn.Conv2d(skip_chs[0], 64, 1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: H/4 -> H/2
        self.up2_conv = nn.Conv2d(64, 128, 3, padding=1)
        self.up2_ps = nn.PixelShuffle(2)  # 128 -> 32 channels
        self.up2_norm = nn.InstanceNorm2d(32)
        self.up2_relu = nn.ReLU(inplace=True)
        self.skip2_proj = nn.Conv2d(skip_chs[1], 32, 1)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: H/2 -> H
        self.up3_conv = nn.Conv2d(32, 64, 3, padding=1)
        self.up3_ps = nn.PixelShuffle(2)  # 64 -> 16 channels
        self.up3_norm = nn.InstanceNorm2d(16)
        self.up3_relu = nn.ReLU(inplace=True)
        self.skip3_proj = nn.Conv2d(skip_chs[2], 16, 1)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final to 3 channels
        self.final = nn.Conv2d(16, 3, 3, padding=1)
    
    def forward(self, z, skips):
        """
        z: [B, 512, H/8, W/8]
        skips: list of [x3, x2, x1] from encoder
        Returns: [B, 3, H, W]
        """
        # Up1: H/8 -> H/4
        d = self.up1_conv(z)
        d = self.up1_ps(d)
        d = self.up1_norm(d)
        d = self.up1_relu(d)
        s = self.skip1_proj(skips[0])
        d = self.fuse1(torch.cat([d, s], dim=1))
        
        # Up2: H/4 -> H/2
        d = self.up2_conv(d)
        d = self.up2_ps(d)
        d = self.up2_norm(d)
        d = self.up2_relu(d)
        s = self.skip2_proj(skips[1])
        d = self.fuse2(torch.cat([d, s], dim=1))
        
        # Up3: H/2 -> H
        d = self.up3_conv(d)
        d = self.up3_ps(d)
        d = self.up3_norm(d)
        d = self.up3_relu(d)
        s = self.skip3_proj(skips[2])
        d = self.fuse3(torch.cat([d, s], dim=1))
        
        return self.final(d)


class AdaptiveWeightPrediction(nn.Module):
    """
    Adaptive weight prediction for trajectory fusion.
    Input: Concat[I_1, I_2, ..., I_T] (3T x H x W)
    Output: w_t(x,y) (T x H x W), sum_t w_t = 1
    """
    def __init__(self, T=10):
        super().__init__()
        self.T = T
        
        self.weight_pred = nn.Sequential(
            nn.Conv2d(3 * T, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.Conv2d(64, T, 3, padding=1)
        )
    
    def forward(self, I_list):
        """
        I_list: list of [B, 3, H, W], length T
        Returns: [B, T, H, W] weights
        """
        x = torch.cat(I_list, dim=1)  # [B, 3T, H, W]
        w = self.weight_pred(x)  # [B, T, H, W]
        w = F.softmax(w, dim=1)  # sum over T = 1
        return w


class ExposureTrajectoryReconstruction(nn.Module):
    """
    Contribution 3: Exposure Trajectory Reconstruction.
    """
    def __init__(self, T=10, feat_ch=512):
        super().__init__()
        self.T = T
        
        # Shared decoder head
        self.decoder = SharedDecoderHead(feat_ch)
        
        # Adaptive weight prediction - input channels = 3 * (T+1) because trajectory has T+1 states
        self.weight_pred = AdaptiveWeightPrediction(T + 1)
    
    def forward(self, trajectory, I_low, skips=None):
        """
        trajectory: list of [B, 512, H/8, W/8], length T+1 (includes initial state Z0)
        I_low: [B, 3, H, W] (for residual connection)
        skips: list of encoder skip features [x3, x2, x1]
        Returns: [B, 3, H, W]
        """
        # Decode all trajectory stages with skip connections
        I_list = [self.decoder(z, skips) for z in trajectory]
        
        # Predict adaptive weights
        w = self.weight_pred(I_list)  # [B, T+1, H, W]
        
        # Weighted fusion
        I_out = 0
        for t in range(len(I_list)):
            I_out = I_out + w[:, t:t+1] * I_list[t]
        
        # Residual connection with input
        I_out = torch.clamp(I_out + I_low, 0, 1)
        
        return I_out


# =============================================================================
# SEF-Net: Main Model
# =============================================================================
class SEFNet(nn.Module):
    """
    SEF-Net: Learning Spatial Exposure Flows for Low-Light Image Enhancement
    A Physics-Informed Exposure State Field Framework
    
    Five-Module Architecture:
    1. Exposure Initialization Encoder
    2. Exposure State Field Constructor
    3. Condition Field Generation
    4. Physics-Informed Spatial Exposure Dynamics
    5. Exposure Trajectory Reconstruction
    """
    def __init__(self,
                 in_ch=3,
                 T=10,
                 C_emb=64,
                 feat_ch=512,
                 ode_steps=10,
                 epsilon=0.1):
        super().__init__()
        
        self.T = T
        self.ode_steps = ode_steps
        
        # Module 1: Exposure Initialization Encoder
        self.encoder = ExposureInitEncoder(in_ch)
        
        # Module 2: Exposure State Field Constructor
        self.exposure_field = ExposureStateFieldConstructor(T, C_emb, feat_ch)
        
        # Module 3: Condition Field Generation
        self.field_heads = ConditionFieldGeneration(feat_ch)
        
        # Module 4: Physics-Informed Spatial Exposure Dynamics
        self.eerf = EERF(feat_ch, epsilon)
        self.ode_func = NeuralODEFunc(feat_ch)
        self.flow_solver = SpatialExposureFlowSolver(
            ode_func=self.ode_func,
            eerf=self.eerf,
            num_steps=ode_steps
        )
        
        # Module 5: Exposure Trajectory Reconstruction
        self.trajectory_recon = ExposureTrajectoryReconstruction(T, feat_ch)
    
    def forward(self, I_low):
        """
        I_low: [B, 3, H, W] in [0, 1]
        Returns:
            I_out: [B, 3, H, W]
            v, n, c: [B, 1, H/8, W/8] - spatial fields
            trajectory: list of exposure states
        """
        # Module 1: Encode
        f0, skips = self.encoder(I_low)  # [B, 512, H/8, W/8], skips=[x3,x2,x1]
        
        # Module 2: Construct Exposure State Field
        Z0_field = self.exposure_field(f0)  # [B, T, 512, H/8, W/8]
        Z0 = Z0_field.mean(dim=1)  # [B, 512, H/8, W/8]  use all T exposure embeddings
        
        # Module 3: Generate Condition Fields
        v, n, c = self.field_heads(f0)  # each [B, 1, H/8, W/8]
        
        # Module 4: Solve Spatial Exposure Flow ODE
        trajectory = self.flow_solver(Z0, v, n, c)
        
        # Module 5: Reconstruct from Trajectory with skip connections
        I_out = self.trajectory_recon(trajectory, I_low, skips)
        
        return I_out, v, n, c, trajectory


# =============================================================================
# Loss Functions
# =============================================================================
class SEFNetLoss(nn.Module):
    """Complete loss function for SEF-Net."""

    def __init__(self, lambda_rec=1.0, lambda_perc=0.1, lambda_phys=0.01,
                 lambda_prior=0.001, lambda_temp=0.01,
                 lambda_color=0.05, lambda_sat=0.05, lambda_contrast=0.05, lambda_tone=0.08,
                 lambda_edge=0.05,
                 epsilon=1e-6, device='cuda'):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_perc = lambda_perc
        self.lambda_phys = lambda_phys
        self.lambda_prior = lambda_prior
        self.lambda_temp = lambda_temp
        self.lambda_color = lambda_color
        self.lambda_sat = lambda_sat
        self.lambda_contrast = lambda_contrast
        self.lambda_tone = lambda_tone
        self.lambda_edge = lambda_edge
        self.epsilon = epsilon

        self.rec_loss_fn = CharbonnierLoss(eps=epsilon)
        try:
            self.perc_loss_fn = PerceptualLoss(device=device)
        except Exception as e:
            print(f"Warning: Could not load PerceptualLoss: {e}")
            self.perc_loss_fn = None
        self.color_loss_fn = ColorLoss()
        self.sat_loss_fn = SaturationLoss()
        self.contrast_loss_fn = ContrastLoss()
        self.tone_loss_fn = ToneLoss()

    def edge_loss(self, pred, target):
        """Gradient (edge) loss using finite differences."""
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    def charbonnier_loss(self, pred, target):
        """Charbonnier loss."""
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))

    def monotonicity_loss(self, trajectory):
        """L_mono = ReLU(Z_k - Z_{k+1})."""
        loss = 0
        for k in range(len(trajectory) - 1):
            loss = loss + F.relu(trajectory[k] - trajectory[k + 1]).mean()
        return loss / (len(trajectory) - 1)

    def exposure_range_loss(self, I_out):
        """L_range = ReLU(I_out - 1) + ReLU(-I_out)."""
        return F.relu(I_out - 1).mean() + F.relu(-I_out).mean()

    def temporal_smoothness_loss(self, I_list):
        """L_temp = |I_{t+1} - I_t|_1."""
        loss = 0
        for t in range(len(I_list) - 1):
            loss = loss + F.l1_loss(I_list[t + 1], I_list[t])
        return loss / (len(I_list) - 1)

    def sep_prior_loss(self, v, I_low):
        """
        Spatial Exposure Principle prior.
        L_sep = ReLU(ρ(v, L) + 0.3), requiring negative correlation.
        """
        # Compute luminance
        L = 0.299 * I_low[:, 0:1] + 0.587 * I_low[:, 1:2] + 0.114 * I_low[:, 2:3]

        # Downsample v and L to same resolution if needed
        if v.shape[2:] != L.shape[2:]:
            v_down = F.adaptive_avg_pool2d(v, L.shape[2:])
        else:
            v_down = v

        # Flatten
        v_flat = v_down.view(-1)
        L_flat = L.view(-1)

        # Compute correlation
        v_mean = v_flat.mean()
        L_mean = L_flat.mean()

        num = ((v_flat - v_mean) * (L_flat - L_mean)).sum()
        den = torch.sqrt(((v_flat - v_mean) ** 2).sum() * ((L_flat - L_mean) ** 2).sum()) + 1e-8

        rho = num / den

        # We want negative correlation, so penalize if rho > -0.3
        return F.relu(rho + 0.3)

    def forward(self, I_out, I_gt, trajectory, v, I_low):
        """
        Compute total loss.
        """
        # Reconstruction loss
        loss_rec = self.rec_loss_fn(I_out, I_gt)

        # Physics losses
        loss_mono = self.monotonicity_loss(trajectory)
        loss_range = self.exposure_range_loss(I_out)
        loss_phys = loss_mono + loss_range

        # SEP prior
        loss_sep = self.sep_prior_loss(v, I_low)

        # Total loss (skip NaN components)
        total_loss = self.lambda_rec * loss_rec
        if not torch.isnan(loss_phys):
            total_loss = total_loss + self.lambda_phys * loss_phys
        if not torch.isnan(loss_sep):
            total_loss = total_loss + self.lambda_prior * loss_sep

        loss_dict = {
            'loss_rec': loss_rec.item(),
            'loss_mono': loss_mono.item() if not torch.isnan(loss_mono) else 0.0,
            'loss_range': loss_range.item() if not torch.isnan(loss_range) else 0.0,
            'loss_sep': loss_sep.item() if not torch.isnan(loss_sep) else 0.0,
        }

        # Perceptual loss
        if self.perc_loss_fn is not None:
            loss_perc = self.perc_loss_fn(I_out, I_gt)
            total_loss = total_loss + self.lambda_perc * loss_perc
            loss_dict['loss_perc'] = loss_perc.item()
        else:
            loss_dict['loss_perc'] = 0.0

        # Color consistency loss
        loss_color = self.color_loss_fn(I_out, I_gt)
        total_loss = total_loss + self.lambda_color * loss_color
        loss_dict['loss_color'] = loss_color.item()

        # Saturation loss
        loss_sat = self.sat_loss_fn(I_out, I_gt)
        total_loss = total_loss + self.lambda_sat * loss_sat
        loss_dict['loss_sat'] = loss_sat.item()

        # Contrast loss
        loss_contrast = self.contrast_loss_fn(I_out, I_gt)
        total_loss = total_loss + self.lambda_contrast * loss_contrast
        loss_dict['loss_contrast'] = loss_contrast.item()

        # Tone loss
        loss_tone = self.tone_loss_fn(I_out, I_gt)
        total_loss = total_loss + self.lambda_tone * loss_tone
        loss_dict['loss_tone'] = loss_tone.item()

        # Edge/gradient loss
        loss_edge = self.edge_loss(I_out, I_gt)
        total_loss = total_loss + self.lambda_edge * loss_edge
        loss_dict['loss_edge'] = loss_edge.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


if __name__ == '__main__':
    # Test
    model = SEFNet(T=10, ode_steps=10)
    I_low = torch.randn(2, 3, 256, 256)
    
    I_out, v, n, c, trajectory = model(I_low)
    
    print(f"I_out shape: {I_out.shape}")
    print(f"v shape: {v.shape}")
    print(f"n shape: {n.shape}")
    print(f"c shape: {c.shape}")
    print(f"Trajectory length: {len(trajectory)}")
    print(f"Trajectory[0] shape: {trajectory[0].shape}")
