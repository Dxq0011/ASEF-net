"""
Loss functions for ASEF-Net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant)"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff ** 2 + self.eps))
        return loss


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, layer='relu3_3', device='cuda'):
        super().__init__()
        import os, torch
        from torchvision import models
        # 优先从本地缓存加载，完全绕过网络请求
        cache_path = os.path.join(
            os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints',
            'vgg16-397923af.pth'
        )
        vgg = models.vgg16(weights=None)
        if os.path.exists(cache_path):
            state = torch.load(cache_path, map_location='cpu', weights_only=True)
            vgg.load_state_dict(state)
            print(f"Loaded VGG16 from local cache: {cache_path}")
        else:
            # 本地没有才走网络
            try:
                from torchvision.models import VGG16_Weights
                vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except ImportError:
                vgg = models.vgg16(pretrained=True)
        vgg = vgg.features.to(device).eval()

        for param in vgg.parameters():
            param.requires_grad = False

        layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }

        self.vgg = vgg
        self.layer_idx = layer_map[layer]
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        pred_feat = self.extract_features(pred)
        target_feat = self.extract_features(target)
        return self.criterion(pred_feat, target_feat)

    def extract_features(self, x):
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i == self.layer_idx:
                return x
        return x


class ColorLoss(nn.Module):
    """Color consistency loss: average color and chroma comparison."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_avg = F.avg_pool2d(pred, kernel_size=pred.shape[2])
        target_avg = F.avg_pool2d(target, kernel_size=target.shape[2])
        pred_chroma = pred - pred.mean(dim=1, keepdim=True)
        target_chroma = target - target.mean(dim=1, keepdim=True)
        return F.mse_loss(pred_avg, target_avg) + F.l1_loss(pred_chroma, target_chroma)


class SaturationLoss(nn.Module):
    """Match target saturation so outputs do not collapse toward gray."""
    def forward(self, pred, target):
        pred_max = pred.max(dim=1, keepdim=True).values
        pred_min = pred.min(dim=1, keepdim=True).values
        target_max = target.max(dim=1, keepdim=True).values
        target_min = target.min(dim=1, keepdim=True).values

        pred_sat = (pred_max - pred_min) / (pred_max + 1e-6)
        target_sat = (target_max - target_min) / (target_max + 1e-6)
        return F.l1_loss(pred_sat, target_sat)


class ContrastLoss(nn.Module):
    """Preserve local luminance contrast against the normal-light target."""
    def forward(self, pred, target):
        device = pred.device
        kernel_x = torch.tensor([[-1., 1.]], device=device).view(1, 1, 1, 2)
        kernel_y = torch.tensor([[-1.], [1.]], device=device).view(1, 1, 2, 1)
        pred_lum = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_lum = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        pred_grad_x = F.conv2d(pred_lum, kernel_x).abs()
        pred_grad_y = F.conv2d(pred_lum, kernel_y).abs()
        target_grad_x = F.conv2d(target_lum, kernel_x).abs()
        target_grad_y = F.conv2d(target_lum, kernel_y).abs()
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)


class ToneLoss(nn.Module):
    """Match luminance distribution and emphasize highlight recovery."""
    def forward(self, pred, target):
        pred_lum = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_lum = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        pred_mean = pred_lum.mean(dim=(2, 3))
        target_mean = target_lum.mean(dim=(2, 3))
        pred_std = pred_lum.std(dim=(2, 3))
        target_std = target_lum.std(dim=(2, 3))

        highlight_weight = 1.0 + 2.0 * (target_lum > 0.75).float()
        tone = F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)
        highlight = torch.mean(highlight_weight * torch.abs(pred_lum - target_lum))
        return tone + highlight


class SmoothLoss(nn.Module):
    """Edge-aware smoothness loss to suppress noise amplification"""
    def __init__(self, delta=5.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, I_low):
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        low_grad_x = torch.abs(I_low[:, :, :, :-1] - I_low[:, :, :, 1:])
        low_grad_y = torch.abs(I_low[:, :, :-1, :] - I_low[:, :, 1:, :])

        weight_x = torch.exp(-self.delta * low_grad_x)
        weight_y = torch.exp(-self.delta * low_grad_y)

        loss = torch.mean(weight_x * pred_grad_x) + torch.mean(weight_y * pred_grad_y)
        return loss


class FieldRegularizationLoss(nn.Module):
    """Regularization to enforce physical meaning of v, n, c"""
    def __init__(self, lambda_v=0.1, lambda_n=0.01, lambda_c=0.01):
        super().__init__()
        self.lambda_v = lambda_v
        self.lambda_n = lambda_n
        self.lambda_c = lambda_c

    def forward(self, v, n, c, I_low):
        luminance = 0.299 * I_low[:, 0:1] + 0.587 * I_low[:, 1:2] + 0.114 * I_low[:, 2:3]
        dark_mask = (luminance < 0.3).float()

        loss_v = self.lambda_v * torch.mean(dark_mask * F.relu(1.0 - v))
        loss_n = self.lambda_n * torch.mean(dark_mask * n)
        color_var = torch.abs(I_low[:, 0:1] - I_low[:, 1:2]) + \
                    torch.abs(I_low[:, 1:2] - I_low[:, 2:3]) + \
                    torch.abs(I_low[:, 0:1] - I_low[:, 2:3])
        loss_c = self.lambda_c * torch.mean(c * color_var)

        return loss_v + loss_n + loss_c


class TotalLoss(nn.Module):
    """Combined loss for ASEF-Net"""
    def __init__(self, 
                 w_rec=1.0, 
                 w_per=0.1, 
                 w_color=0.05,
                 w_smooth=0.01,
                 w_field=0.01,
                 w_sat=0.05,
                 w_contrast=0.05,
                 w_tone=0.08,
                 device='cuda'):
        super().__init__()
        self.w_rec = w_rec
        self.w_per = w_per
        self.w_color = w_color
        self.w_smooth = w_smooth
        self.w_field = w_field
        self.w_sat = w_sat
        self.w_contrast = w_contrast
        self.w_tone = w_tone

        self.rec_loss = CharbonnierLoss()
        self.per_loss = PerceptualLoss(device=device)
        self.color_loss = ColorLoss()
        self.smooth_loss = SmoothLoss()
        self.field_loss = FieldRegularizationLoss()
        self.sat_loss = SaturationLoss()
        self.contrast_loss = ContrastLoss()
        self.tone_loss = ToneLoss()

    def forward(self, pred, target, v, n, c, I_low):
        loss = 0
        loss += self.w_rec * self.rec_loss(pred, target)
        loss += self.w_per * self.per_loss(pred, target)
        loss += self.w_color * self.color_loss(pred, target)
        loss += self.w_smooth * self.smooth_loss(pred, I_low)
        loss += self.w_field * self.field_loss(v, n, c, I_low)
        loss += self.w_sat * self.sat_loss(pred, target)
        loss += self.w_contrast * self.contrast_loss(pred, target)
        loss += self.w_tone * self.tone_loss(pred, target)
        return loss
