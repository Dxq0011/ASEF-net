"""Paired bootstrap confidence intervals for per-image metric differences.

Implements the protocol of Sec. 4.11: 10,000 paired bootstrap repetitions over
per-image scores exported by evaluate.py. For CUED, resampling is clustered by
construction-site identifier (sampling sites with replacement, keeping each
sampled site's images together) to reflect within-site dependence; for LOL-v1,
image-level resampling is used. Intervals are percentile 95% CIs.

The Q4-Q1 contrast tests the difference between group-specific gains and
requires the assignment CSV written by scripts/stratify.py.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

METRICS = ('psnr', 'ssim', 'lpips')


def read_scores(path: str, metric: str):
    with open(path, newline='', encoding='utf8') as fh:
        rows = list(csv.DictReader(fh))
    if any(r[metric] == '' for r in rows):
        raise ValueError(f'metric {metric} missing in {path}; rerun evaluate.py without --no-lpips')
    return {r['image_id']: (float(r[metric]), r.get('site_id', '')) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True, help='per-image CSV of ASEF-Net')
    ap.add_argument('--baseline', required=True, help='per-image CSV of the compared method')
    ap.add_argument('--metric', default='psnr', choices=METRICS)
    ap.add_argument('--reps', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--cluster', action='store_true',
                    help='resample site clusters instead of individual images (CUED)')
    ap.add_argument('--assignment-csv', help='group assignment from stratify.py for the Q4-Q1 contrast')
    ap.add_argument('--out', default='results/bootstrap.json')
    args = ap.parse_args()

    ours = read_scores(args.ours, args.metric)
    base = read_scores(args.baseline, args.metric)
    shared = sorted(set(ours) & set(base))
    if not shared:
        raise ValueError('no shared image ids between the two CSVs')
    a = np.array([ours[k][0] for k in shared])
    b = np.array([base[k][0] for k in shared])
    clusters = {}
    for i, k in enumerate(shared):
        clusters.setdefault(ours[k][1] or k, []).append(i)
    units = [np.array(v) for v in clusters.values()] if args.cluster \
        else [np.array([i]) for i in range(len(shared))]
    unit_of = np.empty(len(shared), dtype=int)
    for u, idx in enumerate(units):
        unit_of[idx] = u

    rng = np.random.default_rng(args.seed)
    n_units = len(units)
    diffs = np.empty(args.reps)
    for r in range(args.reps):
        pick = rng.integers(0, n_units, n_units)
        idx = np.concatenate([units[u] for u in pick])
        diffs[r] = a[idx].mean() - b[idx].mean()
    point = float(a.mean() - b.mean())
    lo, hi = np.percentile(diffs, (2.5, 97.5))
    report = {'metric': args.metric, 'pairs': len(shared), 'reps': args.reps,
              'seed': args.seed, 'resampling': 'site-cluster' if args.cluster else 'image',
              'mean_delta': point, 'ci95': [float(lo), float(hi)],
              'ours': args.ours, 'baseline': args.baseline}

    if args.assignment_csv:
        with open(args.assignment_csv, newline='', encoding='utf8') as fh:
            group = {r['image_id']: r['group'] for r in csv.DictReader(fh)}
        in_q = {q: np.array([group.get(k) == q for k in shared]) for q in ('Q1', 'Q4')}
        for q, flags in in_q.items():
            if not flags.any():
                raise ValueError(f'no shared images in {q}')
        gains = {q: float(a[flags].mean() - b[flags].mean()) for q, flags in in_q.items()}
        boots = np.empty(args.reps)
        for r in range(args.reps):
            pick = rng.integers(0, n_units, n_units)
            idx = np.concatenate([units[u] for u in pick])
            mask = np.zeros(len(shared), dtype=bool)
            mask[idx] = True
            boots[r] = ((a[mask & in_q['Q4']].mean() - b[mask & in_q['Q4']].mean())
                        - (a[mask & in_q['Q1']].mean() - b[mask & in_q['Q1']].mean()))
        qlo, qhi = np.percentile(boots, (2.5, 97.5))
        report['Q4_minus_Q1'] = {'mean_delta': gains['Q4'] - gains['Q1'],
                                 'ci95': [float(qlo), float(qhi)],
                                 'Q1_gain': gains['Q1'], 'Q4_gain': gains['Q4'],
                                 'Q1_n': int(in_q['Q1'].sum()), 'Q4_n': int(in_q['Q4'].sum())}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
