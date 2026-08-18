#!/usr/bin/env python3
"""
Extract per-cluster Fisher enrichment statistics from RoL output.

Replicates the enrichment analysis from step 10 (figure generation) and
prints per-cluster tables with Fisher exact test p-values for:
  - Role × Ecotype (Source Benthic, Source Limnetic, Recipient Benthic,
    Recipient Limnetic)
  - Sex (Male/Female) — only for sex_year_lake stratification
  - Infection (Infected/Non-infected) — only for infection_year_lake

Usage:
    python scripts/extract_enrichments.py [--output-dir OUTPUT_DIR]

Output: Per-stratification tables suitable for manuscript tables.
"""

import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import fisher_exact
import re


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _hypergeom_enrichment(in_cluster, cluster_size, bg_size, bg_hit):
    """Upper-tail hypergeometric enrichment p-value.

    Mirror of surge/analysis.py: Fisher's exact conditions on BOTH margins, so
    a cluster that is exactly the full set of a trait's lakes (in_other == 0)
    — the STRONGEST enrichment — reports p = 1.0.  Conditioning only on the
    cluster size and the background trait count gives it the tiny probability
    it deserves.
    """
    from scipy.stats import hypergeom
    if in_cluster <= 0 or bg_hit <= 0 or cluster_size <= 0:
        return 1.0
    return float(hypergeom.sf(in_cluster - 1, bg_size, bg_hit, cluster_size))


def stratification_from_key(key):
    """Infer stratification type from key format."""
    key_str = str(key)
    if re.search(r'\)-[fFmM]$', key_str):
        return 'sex_year_lake'
    if re.search(r'\)-[01]$', key_str):
        return 'infection_year_lake'
    if re.search(r'\(\d{4}(?:\.\d+)?\)$', key_str):
        return 'year_lake'
    return 'lake'


class EnrichmentAnalyzer:
    """Computes per-cluster enrichments from embeddings and data."""

    def __init__(self, embeddings, data=None):
        self.embeddings = embeddings
        self.keys = sorted(embeddings.keys())
        self.data = data

    def for_stratification(self, strat_type):
        filtered = {k: self.embeddings[k] for k in self.keys
                    if stratification_from_key(k) == strat_type}
        return EnrichmentAnalyzer(filtered, data=self.data)

    def cluster_enrichments(self, n_clusters=4):
        """Cut dendrogram and compute per-cluster Fisher enrichments."""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        X = np.vstack([self.embeddings[k] for k in self.keys])
        Z = linkage(X, method='ward')
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        key_to_cluster = dict(zip(self.keys, labels))

        clusters = defaultdict(list)
        for key, cid in key_to_cluster.items():
            clusters[cid].append(key)

        result = {'n_clusters': n_clusters, 'clusters': {}}
        for cid in sorted(clusters.keys()):
            keys = clusters[cid]
            n = len(keys)
            info = {'size': n, 'keys_sample': keys[:3]}

            # --- Role × Ecotype ---
            if self.data:
                # Unique-LAKE background (not per-key): self.keys holds every
                # stratification of the same lake (year/sex/infection), so
                # counting keys double-counts each lake up to 3× and inflates
                # the enrichment tables.  Mirror surge/analysis.py.
                cluster_lakes = sorted(
                    {k.split(' (')[0] if ' (' in k else k for k in keys})
                all_lakes = sorted(
                    {k.split(' (')[0] if ' (' in k else k for k in self.keys})
                n_lakes = len(cluster_lakes)
                n_bg = len(all_lakes)

                combo_counts = Counter(
                    f'{self.data.get_lake_role(l)} '
                    f'{self.data.get_lake_ecotype(l)}' for l in cluster_lakes)
                bg_combos = Counter(
                    f'{self.data.get_lake_role(l)} '
                    f'{self.data.get_lake_ecotype(l)}' for l in all_lakes)

                info['role_ecotype'] = {}
                for combo in ['Source Benthic', 'Source Limnetic',
                              'Recipient Benthic', 'Recipient Limnetic']:
                    in_cluster = combo_counts.get(combo, 0)
                    if in_cluster == 0:
                        continue
                    in_other = bg_combos.get(combo, 0) - in_cluster
                    # Hypergeometric upper-tail, NOT Fisher: the fully-
                    # contained case (in_other == 0) is the STRONGEST
                    # enrichment — Fisher conditions on both margins and
                    # returns p = 1.0 there.  Mirror surge/analysis.py.
                    p = _hypergeom_enrichment(
                        in_cluster, n_lakes, n_bg, bg_combos.get(combo, 0))
                    info['role_ecotype'][combo] = {
                        'count': in_cluster,
                        'pct': in_cluster / n_lakes,
                        'fisher_p': p,
                    }

                # --- Sex ---
                # The enrichment universe is the SEX-SUFFIXED keys only: a
                # year_lake or infection-suffixed key cannot be "not male", so
                # counting it in the background biased the 2×2.  Mirror
                # surge/analysis.py.
                sexes = []
                for k in keys:
                    if str(k).endswith('-f'):
                        sexes.append('Female')
                    elif str(k).endswith('-m'):
                        sexes.append('Male')
                if sexes:
                    sex_keys_all = [k for k in self.keys
                                    if str(k).endswith('-f')
                                    or str(k).endswith('-m')]
                    bg_sex = Counter(
                        'Female' if str(k).endswith('-f') else 'Male'
                        for k in sex_keys_all)
                    n_sex = len(sex_keys_all)
                    n_sex_cluster = len(sexes)
                    info['sex'] = {}
                    for s in ['Male', 'Female']:
                        in_cluster = sexes.count(s)
                        in_other = bg_sex.get(s, 0) - in_cluster
                        not_in_cluster = n_sex_cluster - in_cluster
                        not_in_other = n_sex - n_sex_cluster - in_other
                        if in_cluster > 0 and in_other > 0:
                            _, p = fisher_exact([[in_cluster, not_in_cluster],
                                                 [in_other, not_in_other]])
                            info['sex'][s] = {
                                'count': in_cluster,
                                'pct': in_cluster / n_sex_cluster,
                                'fisher_p': float(p),
                            }

                # --- Infection ---
                # Same suffixed-keys-only universe as sex (see above).
                infs = []
                for k in keys:
                    m = re.search(r'\)-([01])$', str(k))
                    if m:
                        infs.append('Infected' if int(m.group(1)) == 1
                                    else 'Non-infected')
                if infs:
                    inf_keys_all = [k for k in self.keys
                                    if re.search(r'\)-([01])$', str(k))]
                    bg_inf = Counter(
                        'Infected' if re.search(r'\)-([01])$', str(k))
                        .group(1) == '1' else 'Non-infected'
                        for k in inf_keys_all)
                    n_inf = len(inf_keys_all)
                    n_inf_cluster = len(infs)
                    info['infection'] = {}
                    for lab in ['Infected', 'Non-infected']:
                        in_cluster = infs.count(lab)
                        in_other = bg_inf.get(lab, 0) - in_cluster
                        not_in_cluster = n_inf_cluster - in_cluster
                        not_in_other = n_inf - n_inf_cluster - in_other
                        if in_cluster > 0 and in_other > 0:
                            _, p = fisher_exact([[in_cluster, not_in_cluster],
                                                 [in_other, not_in_other]])
                            info['infection'][lab] = {
                                'count': in_cluster,
                                'pct': in_cluster / n_inf_cluster,
                                'fisher_p': float(p),
                            }

            # --- Year distribution ---
            years = []
            for k in keys:
                if ' (' in k:
                    yr_str = k.split('(')[1].split(')')[0].split('-')[0]
                    try:
                        years.append(int(float(yr_str)))
                    except ValueError:
                        pass
            if years:
                info['year_range'] = f'{min(years)}-{max(years)}'
                info['year_mean'] = float(np.mean(years))

            result['clusters'][cid] = info

        return result


def stars(p):
    if p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def print_enrichment_table(enrichments, strat_name):
    """Pretty-print enrichment results for one stratification."""
    print()
    print("=" * 90)
    print(f"  {strat_name}")
    print("=" * 90)

    for cid in sorted(enrichments['clusters'].keys()):
        ci = enrichments['clusters'][cid]
        print(f"\n  Cluster {cid}  (n = {ci['size']} keys)")

        # Year info
        if 'year_range' in ci:
            print(f"    Years: {ci['year_range']}  (mean = {ci['year_mean']:.1f})")

        # Role × Ecotype
        if 'role_ecotype' in ci:
            print(f"    {'Role × Ecotype':30s} {'Count':>6s} {'%':>6s}  {'p-value':>8s}  Sig")
            print(f"    {'─' * 65}")
            for combo in ['Source Benthic', 'Source Limnetic',
                          'Recipient Benthic', 'Recipient Limnetic']:
                ed = ci['role_ecotype'].get(combo)
                if ed:
                    s = stars(ed['fisher_p'])
                    print(f"    {combo:30s} {ed['count']:4d}/{ci['size']:<4d} "
                          f"{ed['pct']:5.0%}  p={ed['fisher_p']:.4f}  {s}")

        # Sex
        if 'sex' in ci:
            print(f"\n    {'Sex':30s} {'Count':>6s} {'%':>6s}  {'p-value':>8s}  Sig")
            print(f"    {'─' * 65}")
            for lab, ed in ci['sex'].items():
                s = stars(ed['fisher_p'])
                print(f"    {lab:30s} {ed['count']:4d}/{ci['size']:<4d} "
                      f"{ed['pct']:5.0%}  p={ed['fisher_p']:.4f}  {s}")

        # Infection
        if 'infection' in ci:
            print(f"\n    {'Infection':30s} {'Count':>6s} {'%':>6s}  {'p-value':>8s}  Sig")
            print(f"    {'─' * 65}")
            for lab, ed in ci['infection'].items():
                s = stars(ed['fisher_p'])
                print(f"    {lab:30s} {ed['count']:4d}/{ci['size']:<4d} "
                      f"{ed['pct']:5.0%}  p={ed['fisher_p']:.4f}  {s}")

        # Sample keys
        if 'keys_sample' in ci:
            print(f"\n    Sample keys: {', '.join(str(k) for k in ci['keys_sample'])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract per-cluster Fisher enrichment statistics.')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Path to pipeline output directory')
    parser.add_argument('--n-clusters', '-k', type=int, default=4,
                        help='Number of clusters (default: 4)')
    parser.add_argument('--joint', action='store_true',
                        help='Use joint training embeddings')
    args = parser.parse_args()

    # Load data — try joint first, fall back to sequential
    data_path = os.path.join(args.output_dir, 'data.pkl')
    emb_path = os.path.join(args.output_dir, f'embeddings{"_joint" if args.joint else ""}.pkl')
    if not os.path.exists(emb_path):
        # Auto-detect: joint training produces embeddings_joint.pkl
        alt = os.path.join(args.output_dir, 'embeddings_joint.pkl'
                           if not args.joint else 'embeddings.pkl')
        if os.path.exists(alt):
            emb_path = alt
        else:
            print(f"ERROR: Embeddings not found at {emb_path} or {alt}")
            sys.exit(1)

    embeddings = load_pickle(emb_path)
    data = load_pickle(data_path) if os.path.exists(data_path) else None

    analyzer = EnrichmentAnalyzer(embeddings, data=data)

    print(f"Loaded {len(embeddings)} embeddings")
    key_groups = defaultdict(list)
    for k in embeddings:
        key_groups[stratification_from_key(k)].append(k)
    print("Stratification breakdown:", {s: len(ks) for s, ks in key_groups.items()})

    for strat in ['year_lake', 'sex_year_lake', 'infection_year_lake']:
        sub = analyzer.for_stratification(strat)
        if not sub.keys:
            print(f"\n  Skipping {strat} — no keys")
            continue
        enrichments = sub.cluster_enrichments(n_clusters=args.n_clusters)
        print_enrichment_table(enrichments, f"{strat} ({len(sub.keys)} keys)")

    print()
    print("=" * 90)
    print("  Extraction complete.")
    print("=" * 90)


if __name__ == '__main__':
    main()
