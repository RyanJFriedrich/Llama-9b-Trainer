"""Rank-geometry analysis over bulk_out shards.

Reconstructs per-position rank-ordered masses (rows are shuffled by design ->
sort each row's 32 probs descending; slot-0 GT rejoins at its true rank).
Fits candidate shapes to the rank-mass curve, overall and conditioned on the
head mass p1.

Reads only fully-written shards (tolerates a partial last file).
"""
import glob
import json

import numpy as np

OUT = 'data_pipeline/rank_geometry.json'
STRIDE = 16  # keep every 16th row for quantile/fit storage

sum_by_rank = None
logsum_by_rank = None
count = 0
samples = []          # subsampled rank-sorted rows
gt_rank_hist = np.zeros(33, dtype=np.int64)  # rank of GT (1..32), 33 = outside
p1_buckets = [0.0, 0.1, 0.3, 0.6, 0.9, 1.01]
bucket_logsum = np.zeros((len(p1_buckets) - 1, 32))
bucket_count = np.zeros(len(p1_buckets) - 1, dtype=np.int64)

files = sorted(glob.glob('data_pipeline/bulk_out/bulk_*.npz'))
print(f'{len(files)} shards', flush=True)
for f in files:
    try:
        d = np.load(f)
        probs = d['teacher_probs'].astype(np.float64)
        mask = d['loss_mask']
    except Exception as e:
        print(f'skipping {f}: {e}', flush=True)
        continue
    rows = probs[mask == 1]
    rows = np.sort(rows, axis=1)[:, ::-1]  # rank order, descending
    n = rows.shape[0]
    count += n
    s = rows.sum(axis=0)
    ls = np.log(np.clip(rows, 1e-12, None)).sum(axis=0)
    sum_by_rank = s if sum_by_rank is None else sum_by_rank + s
    logsum_by_rank = ls if logsum_by_rank is None else logsum_by_rank + ls
    samples.append(rows[::STRIDE])
    p1 = rows[:, 0]
    bi = np.clip(np.digitize(p1, p1_buckets) - 1, 0, len(p1_buckets) - 2)
    for b in range(len(p1_buckets) - 1):
        sel = bi == b
        bn = int(sel.sum())
        if bn:
            bucket_logsum[b] += np.log(np.clip(rows[sel], 1e-12, None)).sum(axis=0)
            bucket_count[b] += bn
    print(f'{f}: {n} rows (total {count})', flush=True)

mean = sum_by_rank / count
geo = np.exp(logsum_by_rank / count)  # geometric mean per rank
S = np.concatenate(samples)

ranks = np.arange(1, 33, dtype=np.float64)
log_r = np.log(ranks)

def fit_log(x):
    """least squares log-mass = c - a*x ; returns a, c, R^2"""
    y = np.log(np.clip(geo, 1e-20, None))
    A = np.vstack([np.ones_like(x), -x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(coef[1]), float(coef[0]), 1 - ss_res / ss_tot

fits = {
    'power_law (mass ~ r^-alpha)': dict(zip(('alpha', 'c', 'R2'), fit_log(log_r))),
    'exponential (mass ~ e^-beta r)': dict(zip(('beta', 'c', 'R2'), fit_log(ranks))),
}
# stretched exponential: log mass = c - beta r^gamma  (grid over gamma)
best = None
for gamma in np.arange(0.2, 1.51, 0.05):
    beta, c, r2 = fit_log(ranks ** gamma)
    if best is None or r2 > best[3]:
        best = (beta, c, gamma, r2)
fits['stretched_exp (mass ~ e^-beta r^gamma)'] = {
    'beta': best[0], 'c': best[1], 'gamma': best[2], 'R2': best[3]}

# conditional fits (power law per p1 bucket)
cond = {}
for b in range(len(p1_buckets) - 1):
    if bucket_count[b] < 1000:
        continue
    g = np.exp(bucket_logsum[b] / bucket_count[b])
    y = np.log(np.clip(g, 1e-20, None))
    A = np.vstack([np.ones(32), -log_r]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    cond[f'p1 in [{p1_buckets[b]},{p1_buckets[b+1]})'] = {
        'n': int(bucket_count[b]), 'alpha': float(coef[1]), 'R2': r2,
        'mean_captured_mass_top32': float(np.clip(g, 0, 1).sum())}

qs = np.quantile(S, [0.25, 0.5, 0.75], axis=0)
result = {
    'n_positions': int(count),
    'mean_prob_by_rank': mean.tolist(),
    'geomean_prob_by_rank': geo.tolist(),
    'p25_by_rank': qs[0].tolist(),
    'median_by_rank': qs[1].tolist(),
    'p75_by_rank': qs[2].tolist(),
    'fits': fits,
    'conditional_power_law_by_p1_bucket': cond,
}
with open(OUT, 'w') as fh:
    json.dump(result, fh, indent=2)
print(json.dumps(fits, indent=2))
print(json.dumps(cond, indent=2))
print(f'wrote {OUT}')
