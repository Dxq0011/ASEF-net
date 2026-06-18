"""
Inference-time visual refinement for enhanced RGB images.
"""
import numpy as np
from PIL import Image, ImageEnhance


def _auto_contrast_luma(rgb, low_clip=1.0, high_clip=99.2, strength=0.55):
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    lo, hi = np.percentile(luma, [low_clip, high_clip])
    if hi <= lo + 1e-6:
        return rgb

    luma_adj = np.clip((luma - lo) / (hi - lo), 0.0, 1.0)
    ratio = (luma_adj + 1e-4) / (luma + 1e-4)
    ratio = 1.0 + strength * (ratio - 1.0)
    arr = np.clip(arr * ratio[..., None], 0.0, 1.0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode='RGB')


def refine_enhanced_image(image, color=1.22, contrast=1.12, sharpness=1.05, luma_strength=0.55):
    """Make ASEF-Net outputs less gray without changing geometry or resolution."""
    refined = _auto_contrast_luma(image, strength=luma_strength)
    refined = ImageEnhance.Color(refined).enhance(color)
    refined = ImageEnhance.Contrast(refined).enhance(contrast)
    refined = ImageEnhance.Sharpness(refined).enhance(sharpness)
    return refined
