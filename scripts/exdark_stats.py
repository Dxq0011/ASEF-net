"""No-reference image statistics for unpaired ExDark evaluations.

Implements the descriptive indicators of Sec. 4.4 / Table exdark on 8-bit
grayscale (luminance) images:
  - entropy: base-2 entropy of the 256-bin intensity histogram,
  - brightness: mean pixel intensity,
  - contrast: standard deviation of pixel intensity.

These statistics are descriptive; entropy can also grow with noise and
brightness has no universal optimum. Inputs are the enhanced images exported
for each method (optionally resized to the protocol resolution first).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from asef.data import EXTENSIONS


def image_stats(path: Path, size: int | None) -> tuple[float, float, float]:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB'), dtype=np.float32)
        if size:
            rgb = np.asarray(im.convert('RGB').resize((size, size),
                                                      Image.Resampling.BILINEAR),
                             dtype=np.float32)
    y = .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]
    hist = np.bincount(np.clip(y, 0, 255).astype(np.uint8).ravel(), minlength=256)
    p = hist / hist.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    return entropy, float(y.mean()), float(y.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='directory of enhanced images')
    ap.add_argument('--label', default=None, help='method label for the report')
    ap.add_argument('--size', type=int, default=256,
                    help='protocol resize for ExDark statistics (0 = keep native size)')
    ap.add_argument('--out', default='results/exdark_stats.json')
    args = ap.parse_args()

    files = sorted(p for p in Path(args.input).iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f'no images in {args.input}')
    vals = [image_stats(p, args.size or None) for p in files]
    n = len(vals)
    report = {'label': args.label or Path(args.input).name, 'images': n,
              'resize': args.size or None,
              'entropy': sum(v[0] for v in vals) / n,
              'brightness': sum(v[1] for v in vals) / n,
              'contrast': sum(v[2] for v in vals) / n}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
