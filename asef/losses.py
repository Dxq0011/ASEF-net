"""Training losses corresponding to ASEF-Net V3.6 equations."""
from __future__ import annotations
import torch
from torch import nn,Tensor
from torch.nn import functional as F


class VGG19Features(nn.Module):
    """Frozen conv1_2, conv2_2, conv3_3, conv4_3 features."""
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except Exception as err:
            raise RuntimeError('VGG-19 ImageNet weights are required for the perceptual loss. '
                               'Ensure torchvision weights can be downloaded or are cached.') from err
        self.features = net[:26].eval()
        for p in self.features.parameters(): p.requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self,x:Tensor):
        x=(x-self.mean)/self.std
        collected=[]
        for i,layer in enumerate(self.features):
            x=layer(x)
            if i in (2,7,16,25): collected.append(x)
        return collected


class ASEFLoss(nn.Module):
    def __init__(self, delta:float=2., tau:float=.3, lambda_char:float=1.,
                 lambda_perc:float=.1,lambda_mono:float=.05,
                 lambda_sep:float=.1,lambda_temp:float=.05,
                 perceptual:bool=True):
        super().__init__()
        self.delta=float(delta);self.tau=float(tau)
        self.weights={'char':lambda_char,'perc':lambda_perc,'mono':lambda_mono,
                      'sep':lambda_sep,'temp':lambda_temp}
        self.vgg=VGG19Features() if perceptual and lambda_perc>0 else None

    @staticmethod
    def luminance(x:Tensor):
        return (x[:,0:1]*.299+x[:,1:2]*.587+x[:,2:3]*.114)

    def forward(self, pred:dict, gt:Tensor, low:Tensor):
        # Forward decoder runs on [0,1]; scale RGB differences to paper's [0,255].
        residual=255.*(pred['output']-gt)
        char=torch.sqrt(residual.square()+self.delta**2).mean()
        perc=char.new_zeros(())
        if self.vgg is not None:
            pred_feat=self.vgg(pred['output'])
            with torch.no_grad(): gt_feat=self.vgg(gt)
            perc=sum((a-b).abs().mean() for a,b in zip(pred_feat,gt_feat))
        images=pred['stages']
        if images.shape[1]>1:
            # Y_t - Y_(t+1) > 0 penalized; do not modify the ODE stages.
            luma=.299*images[:,:,0]+.587*images[:,:,1]+.114*images[:,:,2]
            mono=F.relu(luma[:,:-1]-luma[:,1:]).abs().mean(dim=(0,2,3)).sum()
            temp=(images[:,1:]-images[:,:-1]).abs().mean(dim=(0,2,3,4)).sum()
        else:
            mono=char.new_zeros(());temp=char.new_zeros(())
        velocity=pred['v_raw'] # retain original map even for mean-field ablation
        low_y=self.luminance(low)
        low_y=F.interpolate(low_y,size=velocity.shape[-2:],mode='area')
        a=velocity.flatten(1)
        b=low_y.flatten(1)
        a=a-a.mean(dim=1,keepdim=True)
        b=b-b.mean(dim=1,keepdim=True)
        norm=torch.linalg.vector_norm(a,dim=1)*torch.linalg.vector_norm(b,dim=1)
        corr=(a*b).sum(dim=1)/norm.clamp_min(1e-8)
        # When either map is exactly constant there is no defined Pearson r.
        corr=torch.where(norm>1e-8,corr,torch.zeros_like(corr))
        sep=F.relu(corr+self.tau).mean()
        components={'char':char,'perc':perc,'mono':mono,'sep':sep,'temp':temp}
        total=sum(self.weights[k]*v for k,v in components.items())
        return total, {**components,'total':total}
