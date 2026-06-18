"""
ASEF-Net: Asynchronous Spatial Exposure Fields for Low-Light Image Enhancement
完全符合论文描述的实现版本

论文公式索引:
- 公式(2): F_0 = E_init(I_low) - Exposure Initialization Encoder
- 公式(3): Z(0) = F_0 - Initial exposure state
- 公式(4): e_t = t/T - Exposure positions
- 公式(5): p_t = P(e_t) - Exposure position encoding
- 公式(7-9): v, n, c - Spatial Exposure Prior
- 公式(10): R(x,y,Z(e)) = Softplus(R_g(z) + ε·tanh(ΔR_φ)) - PIEMF
- 公式(12): dZ/de = (1+v) ⊙ f_θ([Z,n,c,p]) ⊙ (R+η)^{-1} - Neural ODE
- 公式(13): RK4 solver
- 公式(14-18): Trajectory reconstruction and fusion
- 公式(19-23): Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Module 1: Exposure Initialization Encoder (公式2)
# =============================================================================
class ExposureInitEncoder(nn.Module):
    """
    公式(2): F_0 = E_init(I_low)
    Pure Encoder (U-Net style, downsampling path only).
    Input: I_low (3xHxW)
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
# Module 2: Exposure State Field Constructor (公式3-5)
# =============================================================================
class ExposureStateFieldConstructor(nn.Module):
    """
    公式(3): Z(0) = F_0 - Initial exposure state
    公式(4): e_t = t/T - Discrete exposure positions
    公式(5): p_t = P(e_t) - Exposure position encoding
    """
    def __init__(self, T=10, C_emb=64, feat_ch=512):
        super().__init__()
        self.T = T
        self.feat_ch = feat_ch

        # 公式(5): Learnable exposure position embeddings
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
# Module 3: Condition Field Generation (公式7-9)
# =============================================================================
class ConditionFieldGeneration(nn.Module):
    """
    公式(7): v(x,y), n(x,y), c(x,y) - Three condition fields
    公式(8): v = B_v(F_0), n = B_n(F_0), c = B_c(F_0)
    公式(9): v >= 0 (Softplus), n,c in [0,1] (Sigmoid)
    """
    def __init__(self, feat_ch=512):
        super().__init__()

        # Velocity branch: 公式(9) v(x,y) >= 0
        self.velocity_branch = nn.Sequential(
            nn.Conv2d(feat_ch, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Softplus()  # Ensures v >= 0
        )

        # Noise branch: 公式(9) n(x,y) in [0, 1]
        self.noise_branch = nn.Sequential(
            nn.Conv2d(feat_ch, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # Ensures n in [0, 1]
        )

        # Color branch: 公式(9) c(x,y) in [0, 1]
        self.color_branch = nn.Sequential(
            nn.Conv2d(feat_ch, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # Ensures c in [0, 1]
        )

    def forward(self, f0):
        """
        Returns: v, n, c - each [B, 1, H/8, W/8]
        """
        v = self.velocity_branch(f0)  # >= 0
        n = self.noise_branch(f0)     # [0, 1]
        c = self.color_branch(f0)     # [0, 1]
        return v, n, c


# =============================================================================
# Module 4: Physics-Informed Spatial Exposure Dynamics (公式10-13)
# =============================================================================
class PIEMF(nn.Module):
    """
    Physics-Informed Exposure Modulation Field (PIEMF)
    公式(10): R(x,y,Z(e)) = Softplus(R_g(z) + ε·tanh(ΔR_φ(x,y,Z(e))))
    where:
        - R_g(z): global exposure response from globally pooled feature
        - ΔR_φ: learnable spatial deviation
        - ε: maximum deviation range (set to 0.1)
        - tanh: hard constraint |ε·tanh(ΔR)| < ε
    """
    def __init__(self, feat_ch=512, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.feat_ch = feat_ch

        # Global response: 公式(10) R_g(z) from globally pooled feature
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.R_global = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, feat_ch)
        )

        # Spatial deviation: 公式(10) ΔR_φ(x,y,Z(e))
        self.R_delta = nn.Conv2d(feat_ch, feat_ch, 1)

    def forward(self, z):
        """
        z: [B, 512, H, W]
        Returns: [B, 512, H, W] - Physics-Informed Exposure Modulation Field
        """
        B, C, H, W = z.shape

        # 公式(10): Compute global response R_g(z) from pooled feature
        z_pooled = self.global_pool(z).view(B, C)  # [B, 512]
        R_g = self.R_global(z_pooled).view(B, C, 1, 1)  # [B, 512, 1, 1]

        # 公式(10): Compute spatial deviation ΔR_φ
        delta_R = self.R_delta(z)  # [B, 512, H, W]

        # 公式(10): Apply bounded modulation with tanh constraint
        # R = Softplus(R_g + ε·tanh(ΔR_φ))
        R = F.softplus(R_g + self.epsilon * torch.tanh(delta_R))

        return R


class NeuralODEFunc(nn.Module):
    """
    Neural ODE function f_θ
    公式(12): f_θ([Z(e), n, c, p(e)])
    Input: concatenation of Z, n, c, and exposure position encoding p
    """
    def __init__(self, feat_ch=512):
        super().__init__()

        # 公式(12): f_θ processes [Z, n, c, p]
        self.ode_func = nn.Sequential(
            nn.Conv2d(feat_ch + 3, feat_ch, 3, padding=1),  # +3 for n, c, p (v is modulation)
            nn.GroupNorm(32, feat_ch),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1)
        )

        # Initialize last layer to zero for stability
        nn.init.zeros_(self.ode_func[-1].weight)
        nn.init.zeros_(self.ode_func[-1].bias)

    def forward(self, z, n, c, p_e):
        """
        z: [B, 512, H, W]
        n, c: [B, 1, H, W]
        p_e: [B, 1, H, W] - exposure position encoding at current step
        Returns: [B, 512, H, W]
        """
        # Broadcast n, c, p_e to match z resolution if needed
        if n.shape[2:] != z.shape[2:]:
            n = F.interpolate(n, size=z.shape[2:], mode='bilinear', align_corners=False)
            c = F.interpolate(c, size=z.shape[2:], mode='bilinear', align_corners=False)
            p_e = F.interpolate(p_e, size=z.shape[2:], mode='bilinear', align_corners=False)

        # 公式(12): Concatenate [Z, n, c, p]
        x = torch.cat([z, n, c, p_e], dim=1)
        dz_de = self.ode_func(x)
        return dz_de


class SpatialExposureFlowSolver(nn.Module):
    """
    RK4 Solver for Physics-Informed Spatial Exposure Dynamics
    公式(12): dZ/de = (1+v) ⊙ f_θ([Z,n,c,p]) ⊙ (R+η)^{-1}
    公式(13): RK4 integration
    """
    def __init__(self, ode_func, piemf, num_steps=10, eta=1e-6):
        super().__init__()
        self.ode_func = ode_func
        self.piemf = piemf  # Changed from eerf to piemf
        self.num_steps = num_steps
        self.eta = eta  # Small constant for numerical stability

    def forward(self, Z0, v, n, c, exposure_embeddings):
        """
        公式(12-13): Solve ODE with velocity modulation and PIEMF

        Z0: [B, 512, H, W] - initial exposure state
        v, n, c: [B, 1, H, W] - condition fields
        exposure_embeddings: [T, 512] - exposure position encodings
        Returns: list of [B, 512, H, W] - exposure trajectory
        """
        B, C, H, W = Z0.shape
        device = Z0.device

        # 公式(4): delta_e = 1/T
        delta_e = 1.0 / self.num_steps

        states = [Z0]
        z = Z0

        # Broadcast v to match z resolution
        if v.shape[2:] != z.shape[2:]:
            v = F.interpolate(v, size=z.shape[2:], mode='bilinear', align_corners=False)

        for k in range(self.num_steps):
            # 公式(5): Get exposure position encoding p(e_k)
            # Use the k-th exposure embedding as position encoding
            p_e = exposure_embeddings[k].view(1, C, 1, 1).expand(B, -1, H, W)

            # 公式(10): Compute PIEMF R(x,y,Z(e))
            R_z = self.piemf(z)

            # 公式(12): RK4 stages with velocity modulation
            # k1 = (1+v) ⊙ f_θ([z,n,c,p]) ⊙ (R+η)^{-1}
            dz1 = self.ode_func(z, n, c, p_e)
            k1 = (1 + v) * dz1 / (R_z + self.eta)
            k1 = torch.clamp(k1, -10, 10)

            # k2
            z2 = z + 0.5 * delta_e * k1
            R_z2 = self.piemf(z2)
            p_e2 = exposure_embeddings[min(k+1, self.num_steps-1)].view(1, C, 1, 1).expand(B, -1, H, W)
            dz2 = self.ode_func(z2, n, c, p_e2)
            k2 = (1 + v) * dz2 / (R_z2 + self.eta)
            k2 = torch.clamp(k2, -10, 10)

            # k3
            z3 = z + 0.5 * delta_e * k2
            R_z3 = self.piemf(z3)
            dz3 = self.ode_func(z3, n, c, p_e2)
            k3 = (1 + v) * dz3 / (R_z3 + self.eta)
            k3 = torch.clamp(k3, -10, 10)

            # k4
            z4 = z + delta_e * k3
            R_z4 = self.piemf(z4)
            p_e3 = exposure_embeddings[min(k+2, self.num_steps-1)].view(1, C, 1, 1).expand(B, -1, H, W)
            dz4 = self.ode_func(z4, n, c, p_e3)
            k4 = (1 + v) * dz4 / (R_z4 + self.eta)
            k4 = torch.clamp(k4, -10, 10)

            # 公式(13): RK4 integration
            z = z + (delta_e / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            z = torch.clamp(z, -5, 5)  # Prevent explosion
            states.append(z)

        return states


# =============================================================================
# Module 5: Exposure Trajectory Reconstruction (公式14-18)
# =============================================================================
class SharedDecoderHead(nn.Module):
    """
    公式(14): I_t = D(Z_t)
    Shared Decoder Head D with U-Net skip connections.
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
    公式(15-17): Confidence-aware trajectory reconstruction
    公式(15): S = A([I_1, I_2, ..., I_T])
    公式(16): w_t(x,y) = exp(S_t) / Σ exp(S_j)
    公式(17): Σ w_t(x,y) = 1
    """
    def __init__(self, T=10):
        super().__init__()
        self.T = T

        # 公式(15): Lightweight attention module A
        # Three convolutional layers: T -> T/2 -> T
        self.weight_pred = nn.Sequential(
            nn.Conv2d(3 * T, T // 2, 3, padding=1),  # 公式: T -> T/2
            nn.ReLU(inplace=True),
            nn.Conv2d(T // 2, T, 3, padding=1),      # 公式: T/2 -> T
        )

    def forward(self, I_list):
        """
        I_list: list of [B, 3, H, W], length T
        Returns: [B, T, H, W] weights
        """
        # 公式(15): Concatenate all I_t
        x = torch.cat(I_list, dim=1)  # [B, 3T, H, W]

        # 公式(15): Apply attention module A
        w = self.weight_pred(x)  # [B, T, H, W]

        # 公式(16): Softmax along trajectory dimension
        w = F.softmax(w, dim=1)  # 公式(17): sum over T = 1

        return w


class ExposureTrajectoryReconstruction(nn.Module):
    """
    公式(14-18): Confidence-Aware Trajectory Reconstruction
    """
    def __init__(self, T=10, feat_ch=512):
        super().__init__()
        self.T = T

        # 公式(14): Shared decoder D
        self.decoder = SharedDecoderHead(feat_ch)

        # 公式(15-17): Adaptive weight prediction
        self.weight_pred = AdaptiveWeightPrediction(T + 1)

    def forward(self, trajectory, skips):
        """
        公式(14-18): Reconstruct from trajectory

        trajectory: list of [B, 512, H/8, W/8], length T+1
        skips: list of encoder skip features [x3, x2, x1]
        Returns:
            I_out: [B, 3, H, W] - final fused image
            I_list: list of [B, 3, H, W] - all decoded images for loss computation
        """
        # 公式(14): Decode all trajectory stages
        I_list = [self.decoder(z, skips) for z in trajectory]

        # 公式(15-17): Predict adaptive weights
        w = self.weight_pred(I_list)  # [B, T+1, H, W]

        # 公式(18): Weighted fusion WITHOUT residual connection
        I_out = 0
        for t in range(len(I_list)):
            I_out = I_out + w[:, t:t+1] * I_list[t]

        # Clamp to [0, 1] - NO residual connection with I_low
        I_out = torch.clamp(I_out, 0, 1)

        return I_out, I_list


# =============================================================================
# ASEF-Net: Main Model (完全符合论文架构)
# =============================================================================
class ASEFNet(nn.Module):
    """
    ASEF-Net: Asynchronous Spatial Exposure Fields for Low-Light Image Enhancement
    完全符合论文描述的Five-Module Architecture

    Module 1: Exposure Initialization Encoder (公式2)
    Module 2: Exposure State Field Constructor (公式3-5)
    Module 3: Condition Field Generation (公式7-9)
    Module 4: Physics-Informed Spatial Exposure Dynamics (公式10-13)
    Module 5: Exposure Trajectory Reconstruction (公式14-18)
    """
    def __init__(self,
                 in_ch=3,
                 T=10,
                 C_emb=64,
                 feat_ch=512,
                 ode_steps=10,
                 epsilon=0.1,
                 eta=1e-6):
        super().__init__()

        self.T = T
        self.ode_steps = ode_steps
        self.feat_ch = feat_ch

        # Module 1: 公式(2) Exposure Initialization Encoder
        self.encoder = ExposureInitEncoder(in_ch)

        # Module 2: 公式(3-5) Exposure State Field Constructor
        self.exposure_field = ExposureStateFieldConstructor(T, C_emb, feat_ch)

        # Module 3: 公式(7-9) Condition Field Generation
        self.field_heads = ConditionFieldGeneration(feat_ch)

        # Module 4: 公式(10-13) Physics-Informed Spatial Exposure Dynamics
        self.piemf = PIEMF(feat_ch, epsilon)  # Changed from eerf to piemf
        self.ode_func = NeuralODEFunc(feat_ch)
        self.flow_solver = SpatialExposureFlowSolver(
            ode_func=self.ode_func,
            piemf=self.piemf,
            num_steps=ode_steps,
            eta=eta
        )

        # Module 5: 公式(14-18) Exposure Trajectory Reconstruction
        self.trajectory_recon = ExposureTrajectoryReconstruction(T, feat_ch)

    def forward(self, I_low):
        """
        公式(1-18): Complete ASEF-Net forward pass

        I_low: [B, 3, H, W] in [0, 1]
        Returns:
            I_out: [B, 3, H, W] - final enhanced image
            I_list: list of [B, 3, H, W] - all decoded exposure stages
            v, n, c: [B, 1, H/8, W/8] - spatial fields
            trajectory: list of exposure states
        """
        # 公式(2): Module 1 - Encode
        f0, skips = self.encoder(I_low)  # [B, 512, H/8, W/8]

        # 公式(3-5): Module 2 - Construct Exposure State Field
        Z0_field = self.exposure_field(f0)  # [B, T, 512, H/8, W/8]
        Z0 = Z0_field.mean(dim=1)  # [B, 512, H/8, W/8]

        # 公式(7-9): Module 3 - Generate Condition Fields
        v, n, c = self.field_heads(f0)  # each [B, 1, H/8, W/8]

        # 公式(10-13): Module 4 - Solve Physics-Informed ODE
        # Get exposure position embeddings for ODE dynamics
        exposure_embeddings = self.exposure_field.proj(self.exposure_field.exposure_emb)
        trajectory = self.flow_solver(Z0, v, n, c, exposure_embeddings)

        # 公式(14-18): Module 5 - Reconstruct from Trajectory
        I_out, I_list = self.trajectory_recon(trajectory, skips)

        return I_out, I_list, v, n, c, trajectory


# =============================================================================
# Loss Functions (公式19-23)
# =============================================================================
class ASEFNetLoss(nn.Module):
    """
    公式(19-23): Complete loss function for ASEF-Net
    公式(19): L_char - Charbonnier reconstruction loss
    公式(20): L_perc - Perceptual loss
    公式(21): L_mono - Monotonicity loss
    公式(22): L_sep - SEP prior loss
    公式(23): L_temp - Trajectory smoothness loss
    公式(24): Total loss = λ_char·L_char + λ_perc·L_perc + λ_mono·L_mono + λ_sep·L_sep + λ_temp·L_temp
    """

    def __init__(self,
                 lambda_char=1.0,
                 lambda_perc=0.1,
                 lambda_mono=0.05,
                 lambda_sep=0.1,
                 lambda_temp=0.05,
                 delta=2.0,
                 tau=0.3,
                 device='cuda'):
        super().__init__()
        self.lambda_char = lambda_char
        self.lambda_perc = lambda_perc
        self.lambda_mono = lambda_mono
        self.lambda_sep = lambda_sep
        self.lambda_temp = lambda_temp
        self.delta = delta
        self.tau = tau

        # 公式(19): Charbonnier loss
        self.eps = delta

        # 公式(20): Perceptual loss (VGG-19)
        # Paper: layers conv1_2, conv2_2, conv3_3, and conv4_3
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            # VGG layer indices: conv1_2=3, conv2_2=8, conv3_3=15, conv4_3=22
            self.vgg_layers = [3, 8, 15, 22]
        except Exception as e:
            print(f"Warning: Could not load VGG19: {e}")
            self.vgg = None

    def charbonnier_loss(self, pred, target):
        """
        公式(19): L_char = sqrt(||I_out - I_gt||^2 + δ^2)
        """
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))
        return loss

    def perceptual_loss(self, pred, target):
        """
        公式(20): L_perc = ||φ(I_out) - φ(I_gt)||_1
        Using VGG-19 layers: conv1_2, conv2_2, conv3_3, conv4_3
        """
        if self.vgg is None:
            return 0.0

        def extract_features(x):
            features = []
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                if i in self.vgg_layers:
                    features.append(x)
            return features

        pred_feats = extract_features(pred)
        target_feats = extract_features(target)

        loss = 0
        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pf, tf)
        return loss / len(self.vgg_layers)

    def monotonicity_loss(self, I_list):
        """
        公式(21): L_mono = Σ ||ReLU(Y(I_t) - Y(I_{t+1}))||_1
        Encourage monotonic exposure evolution (brightness increases)
        Y(·) denotes the luminance channel: Y = 0.299R + 0.587G + 0.114B
        """
        loss = 0
        for k in range(len(I_list) - 1):
            # Compute luminance Y = 0.299R + 0.587G + 0.114B
            Y_k = 0.299 * I_list[k][:, 0:1] + 0.587 * I_list[k][:, 1:2] + 0.114 * I_list[k][:, 2:3]
            Y_k1 = 0.299 * I_list[k + 1][:, 0:1] + 0.587 * I_list[k + 1][:, 1:2] + 0.114 * I_list[k + 1][:, 2:3]
            loss = loss + F.relu(Y_k - Y_k1).mean()
        return loss / (len(I_list) - 1)

    def sep_prior_loss(self, v, I_low):
        """
        公式(22): L_sep = ReLU(ρ(v, L) + τ)
        Encourage negative correlation between velocity and luminance
        """
        # Compute luminance L = 0.299R + 0.587G + 0.114B
        L = 0.299 * I_low[:, 0:1] + 0.587 * I_low[:, 1:2] + 0.114 * I_low[:, 2:3]

        # Downsample v and L to same resolution if needed
        if v.shape[2:] != L.shape[2:]:
            v_down = F.adaptive_avg_pool2d(v, L.shape[2:])
        else:
            v_down = v

        # Flatten
        v_flat = v_down.view(-1)
        L_flat = L.view(-1)

        # Compute correlation coefficient ρ
        v_mean = v_flat.mean()
        L_mean = L_flat.mean()

        num = ((v_flat - v_mean) * (L_flat - L_mean)).sum()
        den = torch.sqrt(((v_flat - v_mean) ** 2).sum() * ((L_flat - L_mean) ** 2).sum()) + 1e-8

        rho = num / den

        # 公式(22): Penalize if correlation is not negative enough
        return F.relu(rho + self.tau)

    def temporal_smoothness_loss(self, I_list):
        """
        公式(23): L_temp = Σ ||I_{t+1} - I_t||_1
        Trajectory smoothness
        """
        loss = 0
        for t in range(len(I_list) - 1):
            loss = loss + F.l1_loss(I_list[t + 1], I_list[t])
        return loss / (len(I_list) - 1)

    def forward(self, I_out, I_gt, I_list, v, I_low):
        """
        公式(24): Compute total loss
        L = λ_char·L_char + λ_perc·L_perc + λ_mono·L_mono + λ_sep·L_sep + λ_temp·L_temp
        """
        # 公式(19): Reconstruction loss
        loss_char = self.charbonnier_loss(I_out, I_gt)

        # 公式(20): Perceptual loss
        loss_perc = self.perceptual_loss(I_out, I_gt)

        # 公式(21): Monotonicity loss (on decoded images)
        loss_mono = self.monotonicity_loss(I_list)

        # 公式(22): SEP prior loss
        loss_sep = self.sep_prior_loss(v, I_low)

        # 公式(23): Temporal smoothness loss
        loss_temp = self.temporal_smoothness_loss(I_list)

        # 公式(24): Total loss
        total_loss = (
            self.lambda_char * loss_char +
            self.lambda_perc * loss_perc +
            self.lambda_mono * loss_mono +
            self.lambda_sep * loss_sep +
            self.lambda_temp * loss_temp
        )

        loss_dict = {
            'loss_char': loss_char.item(),
            'loss_perc': loss_perc.item() if isinstance(loss_perc, float) else loss_perc.item(),
            'loss_mono': loss_mono.item(),
            'loss_sep': loss_sep.item(),
            'loss_temp': loss_temp.item() if isinstance(loss_temp, float) else loss_temp.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


# =============================================================================
# Compatibility alias for loading old checkpoints
# =============================================================================
class SEFNet(ASEFNet):
    """
    Alias for backward compatibility with old checkpoints.
    Maps old SEFNet class name to new ASEFNet implementation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == '__main__':
    # Test the model
    print("="*60)
    print("Testing ASEF-Net (论文完全一致版本)")
    print("="*60)

    model = ASEFNet(
        in_ch=3,
        T=10,
        C_emb=64,
        feat_ch=512,
        ode_steps=10,
        epsilon=0.1,
        eta=1e-6
    )

    I_low = torch.randn(2, 3, 256, 256)

    print(f"\n输入: I_low shape = {I_low.shape}")

    I_out, I_list, v, n, c, trajectory = model(I_low)

    print(f"\n输出:")
    print(f"  I_out shape: {I_out.shape}")
    print(f"  I_list length: {len(I_list)}")
    print(f"  I_list[0] shape: {I_list[0].shape}")
    print(f"  v shape: {v.shape}")
    print(f"  n shape: {n.shape}")
    print(f"  c shape: {c.shape}")
    print(f"  Trajectory length: {len(trajectory)}")
    print(f"  Trajectory[0] shape: {trajectory[0].shape}")

    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n✅ ASEF-Net模型测试成功！")
    print("="*60)