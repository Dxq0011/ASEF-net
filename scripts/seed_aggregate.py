"""Aggregate repeated-seed evaluations into mean +- sample standard deviation.

Implements the protocol of Sec. 4.11: independently trained seeds (e.g. 2026,
2027, 2028) are each evaluated with evaluate.py; this script reports the
across-seed mean and sample standard deviation (ddof=1) of the per-seed
test-set averages. This describes variation across training runs and is
distinct from the test-image bootstrap intervals of scripts/bootstrap.py.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

METRICS = ('psnr', 'ssim', 'lpips')


def test_set_means(path: str):
    with open(path, newline='', encoding='utf8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f'empty CSV: {path}')
    out = {}
    for m in METRICS:
        vals = [float(r[m]) for r in rows if r[m] != '']
        if vals:
            out[m] = sum(vals) / len(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csvs', nargs='+', required=True,
                    help='one per-image metrics CSV per trained seed, in order')
    ap.add_argument('--seeds', nargs='+', type=int,
                    help='seed identifiers matching --csvs (default: 2026 2027 2028)')
    ap.add_argument('--variant', default='Full ASEF-Net')
    ap.add_argument('--out', default='results/seed_aggregate.json')
    args = ap.parse_args()
    seeds = args.seeds or [2026, 2027, 2028][:len(args.csvs)]
    if len(seeds) != len(args.csvs):
        raise ValueError('provide one seed per CSV')
    if len(seeds) < 2:
        raise ValueError('at least two seeds are required for a sample std')

    per_seed = {s: test_set_means(c) for s, c in zip(seeds, args.csvs)}
    summary = {}
    for m in METRICS:
        vals = [per_seed[s][m] for s in seeds if m in per_seed[s]]
        if len(vals) == len(seeds):
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            summary[m] = {'mean': mean, 'std': var ** .5}

    report = {'variant': args.variant,
              'seeds': seeds,
              'per_seed_test_means': {str(s): per_seed[s] for s in seeds},
              'mean_pm_std': {m: f"{v['mean']:.3f} +/- {v['std']:.3f}"
                              for m, v in summary.items()}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
