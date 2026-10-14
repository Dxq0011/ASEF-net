"""Train ASEF-Net on paired image data using only the declared training partition."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from asef.data import PairedImages
from asef.losses import ASEFLoss
from asef.utils import init_seed, make_device, make_model, load_config,save_checkpoint,load_checkpoint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--data-root',help='override dataset.root in config')
    p.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--device',default='auto')
    p.add_argument('--resume',help='resume optimizer/model from checkpoint')
    p.add_argument('--output',help='override training output directory')
    p.add_argument('--no-vgg',action='store_true',help='quick execution test only, removes VGG perceptual term')
    args=p.parse_args()
    cfg=load_config(args.config)
    train_cfg=cfg['training'];data_cfg=cfg['dataset']
    if args.data_root:data_cfg['root']=args.data_root
    if args.epochs:train_cfg['epochs']=args.epochs
    if args.batch_size:train_cfg['batch_size']=args.batch_size
    init_seed(train_cfg.get('seed',2026))
    device=make_device(args.device)
    model=make_model(cfg).to(device)
    ds=PairedImages(data_cfg['root'],'train',size=data_cfg.get('size',512),
                    augment=data_cfg.get('augment',True),
                    directory_names=data_cfg.get('directory_names'))
    dl=DataLoader(ds,batch_size=train_cfg.get('batch_size',4),shuffle=True,
                  num_workers=train_cfg.get('workers',4),pin_memory=device.type=='cuda')
    weights=cfg.get('loss',{})
    loss_mod=ASEFLoss(delta=weights.get('delta',2.),tau=weights.get('tau',.3),
                      lambda_char=weights.get('char',1.),lambda_perc=0. if args.no_vgg else weights.get('perc',.1),
                      lambda_mono=weights.get('mono',.05),lambda_sep=weights.get('sep',.1),
                      lambda_temp=weights.get('temp',.05),perceptual=not args.no_vgg).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=train_cfg.get('lr',1e-4))
    out=Path(args.output or train_cfg.get('output','runs/asef'))
    out.mkdir(parents=True,exist_ok=True)
    start=0
    if args.resume:
        ckpt=load_checkpoint(model,args.resume,device)
        if 'optimizer' in ckpt:optimizer.load_state_dict(ckpt['optimizer'])
        start=int(ckpt['epoch'])+1
    n_epochs=train_cfg.get('epochs',300)
    for ep in range(start,n_epochs):
        model.train()
        lr=train_cfg.get('lr',1e-4)*(0.5**(ep//100))
        for g in optimizer.param_groups:g['lr']=lr
        running=0.
        for batch in dl:
            low=batch['low'].to(device,non_blocking=True)
            high=batch['high'].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred=model(low)
            loss,parts=loss_mod(pred,high,low)
            if not torch.isfinite(loss).all():raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
            optimizer.step()
            running+=float(loss.detach())
        row={'epoch':ep+1,'train_loss':running/max(len(dl),1),'lr':lr,
             'train_images':len(ds),'seed':train_cfg.get('seed',2026)}
        print(json.dumps(row))
        with (out/'train_metrics.jsonl').open('a',encoding='utf8') as fh:fh.write(json.dumps(row)+'\n')
        save_checkpoint(out/'last.pt',model,optimizer,ep,cfg)
        if (ep+1)%train_cfg.get('save_every',100)==0:
            save_checkpoint(out/f'epoch_{ep+1:04d}.pt',model,optimizer,ep,cfg)
    print(f'Finished training. Parameters: {model.parameters_millions():.3f} M')

if __name__=='__main__':main()
