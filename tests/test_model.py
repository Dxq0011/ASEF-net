import torch
from asef.model import ASEFNet,ASEFConfig


def small(stages=2, steps=1, variant='full', ckpt=False):
    return ASEFNet(ASEFConfig(channels=16,base=2,field_hidden=4,
                dynamics_hidden=8,exposure_emb=4,attention_hidden=8,
                stages=stages,rk4_steps=steps,variant=variant,gradient_checkpointing=ckpt))


def test_forward_shapes_and_exact_rk4_calls():
    model=small(2,2).eval()
    x=torch.rand(1,3,32,32)
    with torch.inference_mode():out=model(x)
    assert out['output'].shape==x.shape
    assert out['stages'].shape==(1,2,3,32,32)
    assert out['v'].shape==(1,1,4,4)
    assert out['weights'].shape==(1,2,1,32,32)
    assert torch.allclose(out['weights'].sum(1),torch.ones_like(out['weights'][:,0]),atol=1e-6)
    assert torch.all(out['v']>=0)
    assert ((out['n']>=0)&(out['n']<=1)).all()
    assert ((out['c']>=0)&(out['c']<=1)).all()
    assert model.count_rhs_calls==4*2*2
    assert torch.isfinite(out['output']).all()


def test_piemf_bound():
    m=small()
    z=torch.randn(1,16,4,4)
    with torch.no_grad():
        response=m.piemf(z)
        shared=torch.nn.functional.softplus(m.piemf.global_response(z))
    assert float((response-shared).abs().max())<m.cfg.epsilon+1e-6


def test_variants_do_not_break_forward():
    for name in ('v-spatial-mean','n-spatial-mean','c-spatial-mean',
                 'no-local-PIEMF','mean-fusion','last-stage-only'):
        model=small(2,1,name).eval()
        with torch.inference_mode():out=model(torch.rand(1,3,32,32))
        assert out['output'].shape==(1,3,32,32)
        assert model.count_rhs_calls==8
        if name=='last-stage-only':assert out['stages'].shape[1]==1
        if name=='v-spatial-mean':
            assert torch.allclose(out['v'],out['v'].mean(dim=(2,3),keepdim=True).expand_as(out['v']))


def test_train_backward_without_checkpoint():
    model=small()
    out=model(torch.rand(1,3,32,32))
    out['output'].mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_train_backward_checkpointed():
    model=small(ckpt=True)
    out=model(torch.rand(1,3,32,32))
    out['output'].mean().backward()
    assert model.encoder.stem[0].weight.grad is not None
