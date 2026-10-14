"""Generate small synthetic train/test pairs to test the complete software pipeline."""
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    root=Path('tests/fixtures/smoke')
    for split in ('train','test'):
        for folder in ('low','high'):
            (root/split/folder).mkdir(parents=True,exist_ok=True)
        for idx in range(2 if split=='train' else 1):
            rng=np.random.default_rng(idx+1)
            high=rng.integers(80,205,size=(32,32,3),dtype=np.uint8)
            low=(high.astype(np.float32)*.28).astype(np.uint8)
            name=f'{idx+1:04d}.png'
            Image.fromarray(low).save(root/split/'low'/name)
            Image.fromarray(high).save(root/split/'high'/name)
    print('Created synthetic smoke-test pairs under tests/fixtures/smoke')

if __name__=='__main__':main()
