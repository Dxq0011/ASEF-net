"""Paired full-reference metrics: RGB PSNR, Gaussian SSIM and LPIPS-VGG."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from torch.nn import functional as F


def psnr(pred:Tensor,gt:Tensor,mask:Tensor|None=None)->Tensor:
    if mask is None:mask=torch.ones_like(pred[:,0:1])
    mask=mask.to(pred.dtype)
    denom=(mask.sum(dim=(1,2,3))*pred.shape[1]).clamp_min(1)
    mse=(((pred-gt)**2)*mask).sum(dim=(1,2,3))/denom
    return -10*torch.log10(mse.clamp_min(1e-12))


def gaussian_ssim(pred:Tensor,gt:Tensor,mask:Tensor|None=None,size:int=11,sigma:float=1.5)->Tensor:
    if min(pred.shape[-2:])<size:raise ValueError('Image too small for 11x11 SSIM')
    k=torch.arange(size,device=pred.device,dtype=pred.dtype)-size//2
    w=torch.exp(-.5*k.square()/sigma**2)
    kernel=(w[:,None]*w[None,:]);kernel=kernel/kernel.sum()
    kernel=kernel.view(1,1,size,size).expand(pred.shape[1],-1,-1,-1)
    conv=lambda x:F.conv2d(x,kernel,groups=pred.shape[1])
    mu_x=conv(pred);mu_y=conv(gt)
    vx=conv(pred**2)-mu_x**2
    vy=conv(gt**2)-mu_y**2
    cov=conv(pred*gt)-mu_x*mu_y
    c1,c2=(.01)**2,(.03)**2
    ssim=((2*mu_x*mu_y+c1)*(2*cov+c2))/((mu_x**2+mu_y**2+c1)*(vx+vy+c2))
    if mask is None:
        return ssim.mean(dim=(1,2,3))
    maskvalid=(F.avg_pool2d(mask.to(pred.dtype),kernel_size=size,stride=1)>=1.0-1e-6)
    denom=(maskvalid.sum(dim=(1,2,3))*pred.shape[1]).clamp_min(1)
    return (ssim*maskvalid).sum(dim=(1,2,3))/denom


class LPIPSVGG(nn.Module):
    def __init__(self):
        super().__init__()
        try:import lpips
        except ImportError as exc:
            raise RuntimeError('Install the lpips Python package for VGG LPIPS evaluation') from exc
        self.metric=lpips.LPIPS(net='vgg',spatial=True).eval()
        for p in self.metric.parameters():p.requires_grad_(False)
    def forward(self, pred:Tensor, gt:Tensor,mask:Tensor|None=None):
        # LPIPS convention: RGB in [-1,1], not raw [0,255].
        spatial=self.metric(2*pred-1,2*gt-1)
        if mask is None: return spatial.mean(dim=(1,2,3))
        mask=F.interpolate(mask.to(spatial.dtype),size=spatial.shape[-2:],mode='nearest')
        return (spatial*mask).sum(dim=(1,2,3))/mask.sum(dim=(1,2,3)).clamp_min(1)
