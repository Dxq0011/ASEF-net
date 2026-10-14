import numpy as np
from PIL import Image
import pytest
from asef.data import PairedImages


def test_pairing_and_masks(tmp_path):
    root=tmp_path/'train'
    for d in ('low','high','valid_masks'):(root/d).mkdir(parents=True)
    for f in ('0001','0002'):
        im=Image.fromarray(np.full((32,32,3),64,dtype=np.uint8))
        im.save(root/'low'/f'{f}.png');im.save(root/'high'/f'{f}.png')
        Image.fromarray(np.full((32,32),255,dtype=np.uint8)).save(root/'valid_masks'/f'{f}.png')
    ds=PairedImages(tmp_path,'train',size=32)
    assert len(ds)==2
    assert ds[0]['low'].shape==(3,32,32)
    assert ds[0]['mask'].shape==(1,32,32)
    (root/'high'/'0002.png').unlink()
    with pytest.raises(ValueError):PairedImages(tmp_path,'train',size=32)
