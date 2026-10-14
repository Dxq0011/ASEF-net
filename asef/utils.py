from __future__ import annotations
import os,random
from pathlib import Path
import yaml
import numpy as np
import torch
from PIL import Image
from .model import ASEFConfig, ASEFNet


def load_config(path:str):
    with open(path,encoding='utf8') as fh:cfg=yaml.safe_load(fh)
    if not isinstance(cfg,dict):raise ValueError('Expected YAML object')
    return cfg


def save_rgb(t:torch.Tensor,path:str|Path):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rgb=(t.detach().float().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)
    Image.fromarray(rgb).save(path)


def make_model(config:dict):
    cfg=ASEFConfig(**config.get('model',{}))
    return ASEFNet(cfg)


def init_seed(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_device(spec:str):
    if spec=='auto':return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(spec)


def load_checkpoint(model:torch.nn.Module,path:str|Path,device):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    state=ckpt.get('model',ckpt) if isinstance(ckpt,dict) else ckpt
    model.load_state_dict(state,strict=True)
    return ckpt


def save_checkpoint(path,model,optimizer,epoch,config):
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'epoch':epoch,'config':config},temp)
    os.replace(temp,path)
