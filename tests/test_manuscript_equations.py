"""Executable checks for equations that can be verified without a trained model.

These checks do not certify the parameter count, historical NFE log or benchmark scores.
"""
from unittest.mock import patch
import torch
from torch import nn
from asef.model import ASEFConfig, ASEFNet
from asef.losses import ASEFLoss


def tiny(stages=2, steps=1):
    return ASEFNet(ASEFConfig(channels=16,base=2,field_hidden=4,
                  dynamics_hidden=8,exposure_emb=4,attention_hidden=8,
                  stages=stages,rk4_steps=steps,gradient_checkpointing=False)).eval()


def test_rk4_queries_exposure_embedding_at_all_intermediate_times():
    model=tiny(stages=2,steps=1)
    times=[]
    orig=model.position.forward
    def trace(e,z):
        times.append(e)
        return orig(e,z)
    with patch.object(model.position,'forward',side_effect=trace):
        with torch.inference_mode(): model(torch.rand(1,3,32,32))
    assert len(times)==8
    assert times==[0.,.25,.25,.5,.5,.75,.75,1.]
    assert model.count_rhs_calls==4*2*1


def test_equation_11_piemf_bound_and_positive_response():
    model=tiny()
    z=torch.randn(2,16,4,4)
    with torch.inference_mode():
        response=model.piemf(z)
        shared=torch.nn.functional.softplus(model.piemf.global_response(z))
    assert (response>0).all()
    assert ((response-shared).abs()<model.cfg.epsilon+1e-5).all()


def test_equations_16_19_logits_one_softmax_and_conv_channels():
    model=tiny(stages=3)
    conv=[m for m in model.attention if isinstance(m,nn.Conv2d)]
    assert len(conv)==3
    assert [(m.in_channels,m.out_channels) for m in conv] == [(9,8),(8,8),(8,3)]
    assert not any(isinstance(m,nn.Softmax) for m in model.attention)
    with torch.inference_mode():out=model(torch.rand(1,3,32,32))
    assert torch.allclose(out['weights'].sum(dim=1),torch.ones_like(out['weights'][:,0]),atol=1e-6)
    assert torch.allclose(out['output'],(out['stages']*out['weights']).sum(dim=1),atol=1e-6)


def test_equations_20_25_charbonnier_reduction_rgb_255_and_weighted_total():
    cfg=ASEFLoss(delta=2.,lambda_perc=0.,perceptual=False)
    pred=torch.tensor([.1,.2,.4],dtype=torch.float32).reshape(1,3,1,1)
    gt=torch.tensor([.2,.2,.0],dtype=torch.float32).reshape(1,3,1,1)
    prediction={'output':pred,'stages':pred.unsqueeze(1),'v_raw':torch.rand(1,1,1,1)}
    loss,parts=cfg(prediction,gt,pred)
    expected=torch.sqrt((255*(pred-gt)).square()+4).mean()
    assert torch.allclose(parts['char'],expected)
    assert torch.allclose(parts['total'],parts['char']+0.1*parts['sep'])
    assert torch.isfinite(loss)


def test_equation_13_channel_concatenation_with_external_velocity_modulation():
    model=tiny()
    z=torch.randn(1,16,4,4)
    v=torch.ones(1,1,4,4)*0.5
    n=torch.full_like(v,0.7)
    c=torch.full_like(v,0.8)
    with torch.inference_mode():
        e=.375
        p=model.position(e,z)
        f=model.dynamics(z,n,c,p)
        r=model.piemf(z)
        target=(1+v)*f/(r+model.cfg.eta)
        actual=model._rhs(e,z,v,n,c)
    assert torch.allclose(actual,target,atol=1e-6,rtol=1e-6)
