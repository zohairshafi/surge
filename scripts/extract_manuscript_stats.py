#!/usr/bin/env python3
"""
Consolidate all manuscript-ready statistics into a single JSON.

Reads every analysis artifact in an output directory and writes ONE
``manuscript_stats.json`` to ``<output-dir>/postprocess/`` so values can be
pulled directly into the manuscript. Covers, for both training paradigms
(sequential and joint) where available:

  1. Dendrogram per-cluster Fisher enrichment (role x ecotype, sex, infection)
  2. Wasserstein Source-vs-Recipient divergence test (t-test, Mann-Whitney)
  3. Mean temporal (Wasserstein) slopes by lake category
  4. PERMANOVA marginal R^2 / p / q per factor per stratification
  5. VQ code GO enrichment (g:Profiler) per comparison
  6. WGCNA eigengene GO enrichment (g:Profiler) per group
  7. Gene-level VQ-code <-> WGCNA-module overlap

Items 5-7 are passed through from ``postprocess/enrichment_summary.json``
(written by the WGCNA pipeline); items 1-4 are extracted here from the raw
``.pkl`` checkpoints. Missing files are skipped gracefully.

Usage:
    python scripts/extract_manuscript_stats.py --output-dir output
"""

import argparse
import json
import os
import pickle
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Reuse the tested dendrogram-Fisher implementation.
from scripts.extract_enrichments import (
    EnrichmentAnalyzer, load_pickle, stratification_from_key)


def _exists(d, name):
    p = os.path.join(d, name)
    return p if os.path.exists(p) else None


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _sanitize(obj):
    """Recursively coerce numpy scalar keys/values to JSON-native types.

    ``json.dump``'s ``default=`` handler only converts *values*, not dict
    *keys* — but cluster IDs from ``scipy.cluster.hierarchy.fcluster`` are
    ``np.int32``, which JSON rejects as a key. This walks the structure once.
    """
    if isinstance(obj, dict):
        out = OrderedDict()
        for k, v in obj.items():
            if isinstance(k, (np.integer,)):
                k = int(k)
            elif isinstance(k, (np.floating,)):
                k = float(k)
            elif not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            out[k] = _sanitize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# 1. Dendrogram Fisher enrichment — both training types, all stratifications
# ---------------------------------------------------------------------------
def extract_dendrogram(out_dir, n_clusters=4):
    """Per-cluster Fisher enrichment for sequential + joint embeddings."""
    data_path = _exists(out_dir, 'data.pkl')
    data = None
    if data_path:
        try:
            data = load_pickle(data_path)
        except Exception as exc:  # e.g. pandas-version mismatch on a pkl
            print(f"  WARNING: data.pkl unreadable ({type(exc).__name__}); "
                  f"role x ecotype enrichment will be skipped.")
            data = None

    out = OrderedDict()
    for variant, suffix in [('sequential', ''), ('joint', '_joint')]:
        emb_path = _exists(out_dir, f'embeddings{suffix}.pkl')
        if emb_path is None:
            continue
        embeddings = load_pickle(emb_path)
        analyzer = EnrichmentAnalyzer(embeddings, data=data)
        per_strat = OrderedDict()
        for strat in ['year_lake', 'sex_year_lake', 'infection_year_lake']:
            sub = analyzer.for_stratification(strat)
            if not sub.keys:
                continue
            per_strat[strat] = sub.cluster_enrichments(n_clusters=n_clusters)
        out[variant] = per_strat
    return out


# ---------------------------------------------------------------------------
# 2 + 3. Wasserstein divergence test + mean slopes
# ---------------------------------------------------------------------------
def extract_wasserstein_slopes(out_dir):
    """Source-vs-Recipient divergence stats + mean slopes per training type.

    Extracts both the pooled (all-strata) stats and the per-stratification
    breakdowns (year_lake, sex_year_lake, infection_year_lake).
    """
    out = OrderedDict()
    for variant, suffix in [('sequential', ''), ('joint', '_joint')]:
        path = _exists(out_dir, f'slopes{suffix}.pkl')
        if path is None:
            continue
        d = load_pickle(path)
        slopes = d.get('slopes', {})
        entry = OrderedDict()
        if slopes:
            entry['n_lakes'] = len(slopes)
            entry['mean_slope'] = float(np.mean(list(slopes.values())))
        for k in ('wasserstein_p95', 'ttest_stat', 'ttest_pvalue',
                  'mannwhitney_stat', 'mannwhitney_pvalue'):
            if d.get(k) is not None:
                entry[k] = float(d.get(k))
        src = d.get('source_slopes', [])
        rec = d.get('recipient_slopes', [])
        if src:
            entry['source_n'] = len(src)
            entry['source_mean_slope'] = float(np.mean(src))
        if rec:
            entry['recipient_n'] = len(rec)
            entry['recipient_mean_slope'] = float(np.mean(rec))

        # Per-stratification breakdown
        per_strat = d.get('stratifications', {})
        if per_strat:
            entry['stratifications'] = {}
            for strat, se in per_strat.items():
                if isinstance(se, dict):
                    entry['stratifications'][str(strat)] = _sanitize(se)

        out[variant] = entry
    return out


# ---------------------------------------------------------------------------
# 4. PERMANOVA
# ---------------------------------------------------------------------------
def extract_permanova(out_dir):
    """Marginal R^2 / p / q per factor, per stratification, per training type."""
    out = OrderedDict()
    for variant, suffix in [('sequential', ''), ('joint', '_joint')]:
        path = _exists(out_dir, f'permanova{suffix}.pkl')
        if path is None:
            continue
        raw = load_pickle(path)
        per_strat = OrderedDict()
        for strat, res in raw.items():
            if not isinstance(res, dict):
                continue
            factors = OrderedDict()
            for factor, d in res.items():
                if not isinstance(d, dict):
                    continue
                factors[str(factor)] = {
                    'r2': float(d.get('r2', d.get('R2', float('nan')))),
                    'p_value': float(d.get('p_value', float('nan'))),
                    'q_value': float(d.get('q_value', d.get('p_value',
                                                             float('nan')))),
                }
            per_strat[str(strat)] = factors
        out[variant] = per_strat
    return out


# ---------------------------------------------------------------------------
# 5-7. g:Profiler enrichment + intersection (pass-through)
# ---------------------------------------------------------------------------
def extract_gprofiler_summary(out_dir):
    """Pass through the WGCNA-pipeline enrichment summary if present."""
    path = _exists(os.path.join(out_dir, 'postprocess'),
                   'enrichment_summary.json')
    if path is None:
        return {'status': 'not_found',
                'note': 'Run wgcna_full / _write_enrichment_summary first.'}
    with open(path) as f:
        comps = json.load(f)

    vq = OrderedDict()
    wgcna = OrderedDict()
    inter = OrderedDict()
    for comp, c in comps.items():
        if not isinstance(c, dict):
            continue
        if c.get('vq_enrichment'):
            vq[comp] = c['vq_enrichment']
        if c.get('eigengene_enrichments'):
            wgcna[comp] = c['eigengene_enrichments']
        if c.get('vq_wgcna_gene_overlap'):
            inter[comp] = c['vq_wgcna_gene_overlap']
    return {
        'vq_code_enrichment': vq,
        'wgcna_enrichment': wgcna,
        'vq_wgcna_intersection': inter,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Consolidate manuscript-ready stats into one JSON.')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Pipeline output directory')
    parser.add_argument('--n-clusters', '-k', type=int, default=4,
                        help='Number of dendrogram clusters (default 4)')
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    post_dir = os.path.join(out_dir, 'postprocess')
    os.makedirs(post_dir, exist_ok=True)

    print(f"[manuscript_stats] Reading from {out_dir}")

    consolidated = OrderedDict()
    consolidated['dendrogram_enrichment'] = extract_dendrogram(
        out_dir, n_clusters=args.n_clusters)
    consolidated['wasserstein_and_slopes'] = extract_wasserstein_slopes(out_dir)
    consolidated['permanova'] = extract_permanova(out_dir)
    consolidated.update(extract_gprofiler_summary(out_dir))

    out_path = os.path.join(post_dir, 'manuscript_stats.json')
    with open(out_path, 'w') as f:
        json.dump(_sanitize(consolidated), f, indent=2)

    # Lightweight console summary
    n_dendo = sum(len(v) for v in
                  consolidated['dendrogram_enrichment'].values())
    n_perm = sum(len(v) for v in consolidated['permanova'].values())
    print(f"  dendrogram_enrichment: {n_dendo} strat-block(s)")
    print(f"  wasserstein_and_slopes: "
          f"{len(consolidated['wasserstein_and_slopes'])} training type(s)")
    print(f"  permanova: {n_perm} strat-block(s)")
    for key in ('vq_code_enrichment', 'wgcna_enrichment',
                'vq_wgcna_intersection'):
        val = consolidated.get(key)
        n = len(val) if isinstance(val, dict) else 0
        status = '' if isinstance(val, dict) else '(not_found)'
        print(f"  {key}: {n} comparison(s) {status}")
    print(f"[manuscript_stats] Wrote {out_path}")


if __name__ == '__main__':
    main()
