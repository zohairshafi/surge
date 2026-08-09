#!/usr/bin/env python3
"""
Export gene lists from infection-associated VQ codes for GO/pathway enrichment.

Reads gene_mappings and embeddings from a RoL pipeline output directory,
identifies VQ codes that are differentially used in infected vs non-infected
strata, and exports the gene lists for those codes.

Usage:
    python scripts/export_infection_genes.py [--output-dir OUTPUT_DIR]

Output: infection_genes_for_go.csv with columns:
    vq_code, p_value, fold_change, infected_mean, noninfected_mean, gene_names
"""

import argparse
import csv
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def infection_code_enrichment(embeddings, codebook_size=None):
    """Identify VQ codes differentially used in infected vs non-infected.

    ``codebook_size`` is derived from the embedding width when not given —
    hard-coding 100 crashed whenever a non-100 codebook was trained.
    """
    inf_keys = [k for k in embeddings if re.search(r'\)-([01])$', str(k))]
    if not inf_keys:
        return {}

    infected = []
    noninfected = []
    for k in inf_keys:
        m = re.search(r'\)-([01])$', str(k))
        is_inf = int(m.group(1)) == 1
        hist = embeddings[k]
        if is_inf:
            infected.append(hist)
        else:
            noninfected.append(hist)

    if not infected or not noninfected:
        return {}

    inf_arr = np.array(infected)
    ninf_arr = np.array(noninfected)
    if codebook_size is None:
        codebook_size = inf_arr.shape[1]

    from scipy.stats import mannwhitneyu

    results = {}
    for code in range(codebook_size):
        inf_usage = inf_arr[:, code]
        ninf_usage = ninf_arr[:, code]
        inf_mean = float(np.mean(inf_usage))
        ninf_mean = float(np.mean(ninf_usage))
        fc = (inf_mean + 1e-8) / (ninf_mean + 1e-8)
        # No blanket try/except → p=1.0; guard only the case Mann-Whitney
        # is genuinely undefined (all values identical across both groups).
        pooled = np.concatenate([inf_usage, ninf_usage])
        if np.all(pooled == pooled[0]):
            print(f"  [export_infection_genes] code {code}: identical usage "
                  f"across all strata — skipping.")
            continue
        _, p = mannwhitneyu(inf_usage, ninf_usage, alternative='two-sided')
        results[code] = {
            'infected_mean': inf_mean,
            'noninfected_mean': ninf_mean,
            'fold_change': fc,
            'p_value': float(p),
        }
    return results


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR q-values (same length/order as input)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Export gene lists from infection-associated VQ codes.')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Path to pipeline output directory')
    parser.add_argument('--joint', action='store_true',
                        help='Use joint training outputs')
    parser.add_argument('--p-threshold', '-p', type=float, default=0.05,
                        help='P-value threshold for significance (default: 0.05)')
    parser.add_argument('--max-genes-per-code', type=int, default=100,
                        help='Max genes to export per code (default: 100)')
    args = parser.parse_args()

    suffix = '_joint' if args.joint else ''

    # Load data
    emb_path = os.path.join(args.output_dir, f'embeddings{suffix}.pkl')
    mappings_path = os.path.join(args.output_dir, f'gene_mappings{suffix}.pkl')

    if not os.path.exists(emb_path):
        print(f"ERROR: Embeddings not found at {emb_path}")
        sys.exit(1)
    if not os.path.exists(mappings_path):
        print(f"ERROR: Gene mappings not found at {mappings_path}")
        sys.exit(1)

    embeddings = load_pickle(emb_path)
    mappings = load_pickle(mappings_path)
    gene_names = mappings.get('gene_names', [])

    print(f"Loaded {len(embeddings)} embeddings, "
          f"{len(mappings.get('gene_to_vq', {}))} mapping keys")

    # Identify infection-associated codes
    results = infection_code_enrichment(embeddings)
    if not results:
        print("ERROR: No infection_year_lake keys found in embeddings")
        sys.exit(1)

    # BH-FDR across all per-code tests — ~100 Mann-Whitney tests with nominal
    # p<0.05 would otherwise report many spurious codes as significant.
    codes_ordered = sorted(results.keys())
    qvals = benjamini_hochberg(
        [results[c]['p_value'] for c in codes_ordered])
    for c, q in zip(codes_ordered, qvals):
        results[c]['q_value'] = float(q)

    sig_codes = [(code, r) for code, r in results.items()
                 if r['q_value'] < args.p_threshold]
    sig_codes.sort(key=lambda x: x[1]['q_value'])

    print(f"Found {len(sig_codes)} significant codes (FDR q < "
          f"{args.p_threshold})")

    if not sig_codes:
        print("No significant codes. Try a higher --p-threshold.")
        sys.exit(0)

    # Build gene lists per code
    out_path = os.path.join(args.output_dir, 'infection_genes_for_go.csv')
    vq_to_gene = mappings.get('vq_to_gene', {})

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vq_code', 'p_value', 'q_value', 'fold_change',
                         'infected_mean', 'noninfected_mean',
                         'n_genes', 'gene_names'])

        for code, r in sig_codes:
            genes = set()
            for key, v2g in vq_to_gene.items():
                if code in v2g:
                    genes.update(v2g[code])

            gene_name_list = []
            for g in sorted(genes)[:args.max_genes_per_code]:
                if gene_names and g < len(gene_names):
                    gene_name_list.append(gene_names[g])
                else:
                    gene_name_list.append(str(g))

            writer.writerow([
                code,
                f"{r['p_value']:.6f}",
                f"{r.get('q_value', float('nan')):.6f}",
                f"{r['fold_change']:.4f}",
                f"{r['infected_mean']:.6f}",
                f"{r['noninfected_mean']:.6f}",
                len(genes),
                ';'.join(gene_name_list),
            ])

    print(f"Exported to {out_path}")
    print()
    print("Top 5 infection-associated codes:")
    for code, r in sig_codes[:5]:
        genes = set()
        for key, v2g in vq_to_gene.items():
            if code in v2g:
                genes.update(v2g[code])
        print(f"  Code {code}: p={r['p_value']:.4f}, "
              f"FC={r['fold_change']:.2f}, "
              f"{len(genes)} genes")
    print()
    print("Next step: paste the gene_name column into g:Profiler, DAVID, or Enrichr.")


if __name__ == '__main__':
    main()
