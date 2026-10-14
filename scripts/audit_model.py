"""Read-only model/PyTorch contract audit; outputs actual model statistics."""
from __future__ import annotations
import argparse,json
from asef.utils import load_config,make_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    cfg=load_config(args.config)
    model=make_model(cfg)
    stages=model.cfg.stages;steps=model.cfg.rk4_steps
    pcount=sum(p.numel() for p in model.parameters())
    print(json.dumps({'model':'ASEFNet','parameters':pcount,'parameters_millions':round(pcount/1e6,6),
                      'channels':model.cfg.channels,'stages':stages,'rk4_steps_per_stage':steps,
                      'standard_RK4_slope_evaluations':4*stages*steps,
                      'attention_conv_layers':sum(1 for m in model.attention.modules() if m.__class__.__name__=='Conv2d'),
                      'perceptual_feature_layers':['conv1_2','conv2_2','conv3_3','conv4_3']},indent=2))

if __name__=='__main__':main()
