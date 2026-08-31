"""Compare two prompt_probabilities captures: accuracy + agreement + overlap.

Usage: headtohead_compare.py <capture_a.json> <capture_b.json> [skip_prefix]
skip_prefix drops the first N positions from both captures (high-entropy head).
"""
import json
import math
import sys

def load(path):
    with open(path, 'rb') as fh:
        d = json.load(fh)
    return d['prompt_probabilities'], d['timings']

def stats(pp):
    nll = 0.0
    top1 = 0
    n = 0
    top10_mass = 0.0
    for e in pp:
        n += 1
        nll -= e['logprob']
        tl = e['top_logprobs']
        top10_mass += sum(math.exp(t['logprob']) for t in tl)
        if tl and tl[0]['id'] == e['id']:
            top1 += 1
    return {
        'n': n,
        'mean_nll': nll / n,
        'ppl': math.exp(nll / n),
        'top1_acc': top1 / n,
        'mean_top10_mass': top10_mass / n,
    }

def cross_stats(pp_a, pp_b):
    assert len(pp_a) == len(pp_b)
    agree = 0
    inter = 0.0
    cover_a = 0.0
    cover_b = 0.0
    n = len(pp_a)
    for ea, eb in zip(pp_a, pp_b):
        A = {t['id']: math.exp(t['logprob']) for t in ea['top_logprobs']}
        B = {t['id']: math.exp(t['logprob']) for t in eb['top_logprobs']}
        if ea['top_logprobs'] and eb['top_logprobs'] and \
           ea['top_logprobs'][0]['id'] == eb['top_logprobs'][0]['id']:
            agree += 1
        common = set(A) & set(B)
        inter += len(common)
        cover_a += sum(A[k] for k in common) / max(sum(A.values()), 1e-9)
        cover_b += sum(B[k] for k in common) / max(sum(B.values()), 1e-9)
    return {
        'top1_agreement': agree / n,
        'top10_intersection': inter / n,
        'mass_A_in_B': cover_a / n,
        'mass_B_in_A': cover_b / n,
    }

skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0
pp_a, ta = load(sys.argv[1])
pp_b, tb = load(sys.argv[2])
pp_a, pp_b = pp_a[skip:], pp_b[skip:]
sa, sb = stats(pp_a), stats(pp_b)
cs = cross_stats(pp_a, pp_b)

if skip:
    print(f'(first {skip} positions stripped; n={sa["n"]})\n')
print(f'{"metric":<22}{"A: " + sys.argv[1].split("/")[-1]:>34}{"B: " + sys.argv[2].split("/")[-1]:>34}')
for k in ('n', 'mean_nll', 'ppl', 'top1_acc', 'mean_top10_mass'):
    print(f'{k:<22}{sa[k]:>34.4f}{sb[k]:>34.4f}')
print(f'{"prefill tok/s":<22}{ta["prompt_per_second"]:>34.1f}{tb["prompt_per_second"]:>34.1f}')
print(f'\ntop-1 agreement between models:      {cs["top1_agreement"]:.4f}')
print(f'mean top-10 intersection:            {cs["top10_intersection"]:.2f} / 10')
print(f'of A top-10 mass, frac also in B:    {cs["mass_A_in_B"]:.4f}')
print(f'of B top-10 mass, frac also in A:    {cs["mass_B_in_A"]:.4f}')
