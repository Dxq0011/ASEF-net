"""
Evaluation metrics for ASEF-Net
"""
import torch
import numpy as np
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:
    peak_signal_noise_ratio = None
    structural_similarity = None


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Ensure range [0, max_val]
    img1 = np.clip(img1, 0, max_val)
    img2 = np.clip(img2, 0, max_val)

    # If batch, compute mean
    if img1.ndim == 4:
        psnrs = []
        for i in range(img1.shape[0]):
            if peak_signal_noise_ratio is None:
                mse = np.mean((img1[i] - img2[i]) ** 2)
                psnr = 100.0 if mse == 0 else 20 * np.log10(max_val / np.sqrt(mse))
            else:
                psnr = peak_signal_noise_ratio(img1[i], img2[i], data_range=max_val)
            psnrs.append(psnr)
        return np.mean(psnrs)
    else:
        if peak_signal_noise_ratio is None:
            mse = np.mean((img1 - img2) ** 2)
            return 100.0 if mse == 0 else 20 * np.log10(max_val / np.sqrt(mse))
        return peak_signal_noise_ratio(img1, img2, data_range=max_val)


def calculate_ssim(img1, img2, max_val=1.0):
    """Calculate SSIM between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    img1 = np.clip(img1, 0, max_val)
    img2 = np.clip(img2, 0, max_val)

    if img1.ndim == 4:
        ssims = []
        for i in range(img1.shape[0]):
            if structural_similarity is None:
                ssims.append(_fallback_ssim(img1[i], img2[i], max_val))
                continue
            # Move channel to last dim for skimage
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            ssim = structural_similarity(im1, im2, data_range=max_val, 
                                          channel_axis=2, multichannel=True)
            ssims.append(ssim)
        return np.mean(ssims)
    else:
        if structural_similarity is None:
            return _fallback_ssim(img1, img2, max_val)
        im1 = np.transpose(img1, (1, 2, 0))
        im2 = np.transpose(img2, (1, 2, 0))
        return structural_similarity(im1, im2, data_range=max_val, 
                                     channel_axis=2, multichannel=True)


def _fallback_ssim(img1, img2, max_val=1.0):
    """Small dependency-free SSIM approximation used when scikit-image is absent."""
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
    )


class LPIPSMetric:
    """LPIPS perceptual distance"""
    def __init__(self, net='alex', device='cuda'):
        import lpips
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.device = device

    def __call__(self, img1, img2):
        """
        img1, img2: [B, 3, H, W] in [0, 1]
        Returns: mean LPIPS distance (lower is better)
        """
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float().to(self.device)
        if isinstance(img2, np.ndarray):
            img2 = torch.from_numpy(img2).float().to(self.device)

        # LPIPS expects [-1, 1]
        img1 = img1 * 2.0 - 1.0
        img2 = img2 * 2.0 - 1.0

        with torch.no_grad():
            dist = self.loss_fn(img1, img2)

        return dist.mean().item()


def calculate_niqe(img):
    """NIQE (Natural Image Quality Evaluator) - no-reference metric"""
    # This requires the pyiqa package or MATLAB implementation
    # Placeholder - recommend using pyiqa
    try:
        import pyiqa
        niqe_metric = pyiqa.create_metric('niqe')
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        with torch.no_grad():
            score = niqe_metric(img)
        return score.mean().item()
    except ImportError:
        print("Warning: pyiqa not installed. NIQE calculation skipped.")
        return 0.0


class MetricTracker:
    """Track metrics during evaluation"""
    def __init__(self, lpips_device='cuda'):
        self.psnrs = []
        self.ssims = []
        self.lpipss = []
        try:
            self.lpips_fn = LPIPSMetric(device=lpips_device)
        except ImportError:
            self.lpips_fn = None

    def update(self, pred, target):
        self.psnrs.append(calculate_psnr(pred, target))
        self.ssims.append(calculate_ssim(pred, target))
        if self.lpips_fn is not None:
            self.lpipss.append(self.lpips_fn(pred, target))

    def get_results(self):
        return {
            'PSNR': np.mean(self.psnrs),
            'SSIM': np.mean(self.ssims),
            'LPIPS': np.mean(self.lpipss) if self.lpipss else 0.0
        }

    def reset(self):
        self.psnrs = []
        self.ssims = []
        self.lpipss = []
