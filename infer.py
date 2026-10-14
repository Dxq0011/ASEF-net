"""Infer enhanced RGB images with the model architecture configured for training."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from asef.utils import load_config,make_model,make_device,load_checkpoint,save_rgb
from asef.data import EXTENSIONS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto')
    args=p.parse_args()
    cfg=load_config(args.config);size=cfg['dataset'].get('size',512)
    device=make_device(args.device)
    model=make_model(cfg).to(device).eval()
    load_checkpoint(model,args.checkpoint,device)
    inp=Path(args.input)
    images=[inp] if inp.is_file() else [p for p in sorted(inp.iterdir()) if p.suffix.lower() in EXTENSIONS]
    if not images:raise ValueError('No input images found')
    for img in images:
        with Image.open(img) as pil:
            original=pil.convert('RGB');w,h=original.size
            sample=original.resize((size,size),Image.Resampling.BILINEAR)
            tensor=torch.from_numpy(np.asarray(sample,dtype='float32').copy().transpose(2,0,1)/255.).unsqueeze(0).to(device)
        with torch.inference_mode(): pred=model(tensor)['output']
        out=torch.nn.functional.interpolate(pred,size=(h,w),mode='bilinear',align_corners=False)
        save_rgb(out[0],Path(args.output)/(img.stem+'.png'))
    print(f'Saved {len(images)} images to {args.output}')

if __name__=='__main__':main()
