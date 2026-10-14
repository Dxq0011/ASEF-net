import torch
from asef.losses import ASEFLoss
from asef.metrics import psnr,gaussian_ssim


def test_charbonnier_pixel_average_and_losses():
    mod=ASEFLoss(perceptual=False,lambda_perc=0)
    pred=torch.full((1,3,16,16),.5)
    gt=torch.full((1,3,16,16),.4)
    stages=torch.stack((pred, pred+.01),dim=1)
    fields={'output':pred,'stages':stages,'v_raw':torch.rand(1,1,2,2)}
    loss,components=mod(fields,gt,torch.rand_like(pred))
    expected=torch.sqrt(torch.tensor((255*.1)**2+2.**2))
    assert torch.isclose(components['char'],expected,atol=1e-4)
    assert torch.isfinite(loss)


def test_masked_metrics():
    p=torch.rand(1,3,24,24)
    mask=torch.ones(1,1,24,24)
    assert float(psnr(p,p,mask)[0])>110
    assert abs(float(gaussian_ssim(p,p,mask)[0])-1)<1e-5
