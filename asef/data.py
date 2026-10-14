"""Paired RGB datasets with independent train/test roots and optional valid masks."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.tiff','.tif'}


def list_image_files(folder:Path):
    if not folder.exists(): raise FileNotFoundError(f'Missing dataset folder: {folder}')
    files={p.stem:p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if not files: raise ValueError(f'No images in {folder}')
    return files


class PairedImages(Dataset):
    """Folders: {split}/low, {split}/high, optionally {split}/valid_masks.

    Pair by exact filename stem. Test images and metadata are never consulted
    during weight optimization or epoch selection.
    """
    def __init__(self,root:str|Path,split:str,size:int=512,augment:bool=False,
                 directory_names:dict|None=None):
        self.root=Path(root)
        self.split=split
        self.size=size
        self.augment=augment
        dirname=(directory_names or {}).get(split,split)
        self.path=self.root/dirname
        low=list_image_files(self.path/'low')
        gt=list_image_files(self.path/'high')
        if low.keys()!=gt.keys():
            a=sorted(low.keys()-gt.keys()); b=sorted(gt.keys()-low.keys())
            raise ValueError(f'Paired dataset mismatch: low-only={a[:5]}, high-only={b[:5]}')
        self.ids=sorted(low)
        self.low=low;self.gt=gt
        maskdir=self.path/'valid_masks'
        self.masks=list_image_files(maskdir) if maskdir.exists() else {}
        extra=set(self.masks)-set(self.ids)
        if extra: raise ValueError(f'Mask(s) have no image match: {sorted(extra)[:5]}')
        self.metadata={}
        csvfile=self.path/'metadata.csv'
        if csvfile.exists():
            import csv
            with csvfile.open(newline='',encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):self.metadata[row['image_id']]=row

    def __len__(self):return len(self.ids)

    @staticmethod
    def load(path:Path,size:int,mode:str):
        with Image.open(path) as im:
            im=im.convert(mode)
            im=im.resize((size,size),Image.Resampling.BILINEAR if mode=='RGB' else Image.Resampling.NEAREST)
            return np.asarray(im,dtype=np.float32).copy()

    def __getitem__(self,i:int):
        name=self.ids[i]
        low=self.load(self.low[name],self.size,'RGB')/255.
        high=self.load(self.gt[name],self.size,'RGB')/255.
        if name in self.masks:
            mask=self.load(self.masks[name],self.size,'L')/255.
            mask=(mask>=.5).astype('float32')
        else:mask=np.ones((self.size,self.size),dtype=np.float32)
        # Paired augmentation with shared transforms; masks follow image.
        if self.augment:
            if random.random()<.5:
                low=np.flip(low,axis=1);high=np.flip(high,axis=1);mask=np.flip(mask,axis=1)
            if random.random()<.5:
                low=np.flip(low,axis=0);high=np.flip(high,axis=0);mask=np.flip(mask,axis=0)
        low=torch.from_numpy(np.ascontiguousarray(low.transpose(2,0,1)))
        high=torch.from_numpy(np.ascontiguousarray(high.transpose(2,0,1)))
        mask=torch.from_numpy(np.ascontiguousarray(mask[None]))
        row=self.metadata.get(name,{})
        return {'low':low,'high':high,'mask':mask,'image_id':name,
                'site_id':row.get('site_id',''), 'split':self.split}
