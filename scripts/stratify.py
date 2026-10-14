"""Illumination-heterogeneity stratification of per-image CUED metrics.

Implements the protocol of Sec. 4.7: the grouping variable

    H(I_low) = P90(Y_low) - P10(Y_low)

is computed from the 8-bit luminance of each low-light input, independently of
references and predictions. Test images are sorted by H and partitioned into
`--groups` disjoint rank-based groups of equal size. Per-group means are
arithmetic averages of the same per-image measurements exported by
evaluate.py, so group means are consistent with the whole-test averages.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
from asef.data import EXTENSIONS

METRICS = ('psnr', 'ssim', 'lpips')


def luminance_span(path: Path, size: int) -> float:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR),
                         dtype=np.float32)
    y = .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]
    p10, p90 = np.percentile(y, (10, 90))
    return float(p90 - p10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='per-image metrics CSV from evaluate.py')
    ap.add_argument('--low-dir', required=True, help='low-light test inputs of the same split')
    ap.add_argument('--groups', type=int, default=4)
    ap.add_argument('--size', type=int, default=512,
                    help='evaluation input size used for the H statistic')
    ap.add_argument('--out', default='results/stratified.json')
    args = ap.parse_args()

    with open(args.csv, newline='', encoding='utf8') as fh:
        rows = list(csv.DictReader(fh))
    low_files = {p.stem: p for p in Path(args.low_dir).iterdir()
                 if p.is_file() and p.suffix.lower() in EXTENSIONS}
    missing = [r['image_id'] for r in rows if r['image_id'] not in low_files]
    if missing:
        raise FileNotFoundError(f'low-light inputs not found for {missing[:5]}')

    spans = {r['image_id']: luminance_span(low_files[r['image_id']], args.size)
             for r in rows}
    order = sorted(rows, key=lambda r: (spans[r['image_id']], r['image_id']))
    n = len(order)
    sizes = [n // args.groups + (1 if i < n % args.groups else 0)
             for i in range(args.groups)]
    bounds, start = [], 0
    for k, sz in enumerate(sizes, start=1):
        members = order[start:start + sz]
        start += sz
        per_metric = {m: [float(r[m]) for r in members if r[m] != '']
                      for m in METRICS}
        summary = {'group': f'Q{k}', 'n': sz,
                   'H_range': [round(min(spans[r['image_id']] for r in members)),
                               round(max(spans[r['image_id']] for r in members))]}
        summary.update({m: sum(v) / len(v) for m, v in per_metric.items() if v})
        bounds.append(summary)

    all_rows = {r['image_id']: r for r in rows}
    overall = {m: sum(float(r[m]) for r in rows if r[m] != '') / n for m in METRICS}
    report = {'csv': args.csv, 'groups': args.groups,
              'per_group': bounds,
              'all': {'n': n, **overall},
              'contrast_Q4_minus_Q1': {
                  m: bounds[-1][m] - bounds[0][m]
                  for m in METRICS if m in bounds[-1] and m in bounds[0]}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    with out.with_suffix('.assignment.csv').open('w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['image_id', 'H', 'group'])
        for k, sz in enumerate(sizes, start=1):
            for r in order[sum(sizes[:k - 1]):sum(sizes[:k])]:
                w.writerow([r['image_id'], f"{spans[r['image_id']]:.4f}", f'Q{k}'])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
