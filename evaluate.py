"""Evaluate fixed checkpoint on paired test images; export one CSV row per image."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from asef.data import PairedImages
from asef.metrics import psnr,gaussian_ssim,LPIPSVGG
from asef.utils import load_config,make_model,make_device,load_checkpoint,save_rgb


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root')
    p.add_argument('--out',default='results/per_image.csv')
    p.add_argument('--device',default='auto')
    p.add_argument('--no-lpips',action='store_true',help='skip LPIPS for quick testing')
    p.add_argument('--save-images',help='optional output directory')
    args=p.parse_args()
    cfg=load_config(args.config)
    data=cfg['dataset'];device=make_device(args.device)
    if args.data_root:data['root']=args.data_root
    model=make_model(cfg).to(device).eval()
    load_checkpoint(model,args.checkpoint,device)
    ds=PairedImages(data['root'],'test',size=data.get('size',512),augment=False,
                    directory_names=data.get('directory_names'))
    dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    lpips_metric=LPIPSVGG().to(device).eval() if not args.no_lpips else None
    rows=[]
    with torch.inference_mode():
        for batch in dl:
            low=batch['low'].to(device)
            high=batch['high'].to(device)
            mask=batch['mask'].to(device)
            pred=model(low)['output']
            a=float(psnr(pred,high,mask)[0]); b=float(gaussian_ssim(pred,high,mask)[0])
            c=float(lpips_metric(pred,high,mask)[0]) if lpips_metric else None
            name=batch['image_id'][0]
            rows.append({'image_id':name,'site_id':batch['site_id'][0],
                         'psnr':a,'ssim':b,'lpips':c,'valid_mask_ratio':float(mask.mean())})
            if args.save_images:save_rgb(pred[0],Path(args.save_images)/(name+'.png'))
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf8') as fh:
        writer=csv.DictWriter(fh,fieldnames=['image_id','site_id','psnr','ssim','lpips','valid_mask_ratio'])
        writer.writeheader();writer.writerows(rows)
    report={'samples':len(rows),'psnr':sum(r['psnr'] for r in rows)/len(rows),
            'ssim':sum(r['ssim'] for r in rows)/len(rows),
            'lpips':sum(r['lpips'] for r in rows)/len(rows) if lpips_metric else None,
            'checkpoint':args.checkpoint}
    print(json.dumps(report,indent=2))
    output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')

if __name__=='__main__':main()
