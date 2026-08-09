#!/usr/bin/env python3
"""
Comprehensive WGCNA pipeline with VQ-GNN comparison.

Runs WGCNA separately on each group within a stratification, performs
two-way module preservation, identifies consensus (preserved) modules,
feeds eigengene-correlated genes into g:Profiler enrichment, and
compares VQ-code gene sets against WGCNA module gene sets.

Follows best practices from the literature:
  - Test between-group differences before analysis
  - Run WGCNA independently per group (never pool blindly)
  - Module preservation both ways (swap reference/test)
  - Consensus via preserved modules
  - Module-level diagnostics, not just global metrics

Comparisons:
  1. Infection:  infected vs non-infected  (2021, pooled across lakes)
  2. Sex:        female vs male            (2021, pooled across lakes)
  3. Year:       2019 vs 2023              (best lake by sample count)

Usage:
    python scripts/wgcna_pipeline.py \
        --matrices rol/output/matrices.pkl \
        --gene-mappings rol/output/gene_mappings.pkl \
        --output-dir rol/output/wgcna \
        --top-n-genes 5000 --min-expression 1e-5
"""

import argparse
import json
import csv
import os
import pickle
import re
import sys
import time
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# ===========================================================================
# g:Profiler API (reused from vq_gprofiler.py / spi1b_intersection_genes.py)
# ===========================================================================

GPROFILER_CONVERT_URL = "https://biit.cs.ut.ee/gprofiler/api/convert/convert/"
GPROFILER_GOST_URL   = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


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


def convert_to_ensembl(genes, organism="gaculeatus"):
    """Convert gene symbols to Ensembl IDs."""
    payload = {"organism": organism, "query": genes, "target": "ENSG"}
    resp = requests.post(GPROFILER_CONVERT_URL, json=payload, timeout=120)
    if resp.status_code != 200:
        return [], {}
    ensembl_ids = []
    ensembl_to_name = {}
    for r in resp.json()["result"]:
        name = r["incoming"]
        c = r.get("converted", "")
        if c and c != "None":
            ensembl_ids.append(c)
            ensembl_to_name[c] = name
    return ensembl_ids, ensembl_to_name


def run_enrichment(ensembl_ids, organism="gaculeatus",
                   sources=None, user_threshold=0.05):
    """Run g:Profiler enrichment. Returns list of term dicts."""
    if sources is None:
        sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "WP", "REAC"]
    payload = {
        "organism": organism,
        "query": ensembl_ids,
        "sources": sources,
        "user_threshold": user_threshold,
        "no_evidences": True,
    }
    resp = requests.post(GPROFILER_GOST_URL, json=payload, timeout=120)
    if resp.status_code != 200:
        return []
    results = []
    for term in resp.json().get("result", []):
        results.append({
            "source": term["source"],
            "term_id": term["native"],
            "term_name": term["name"],
            "p_value": term["p_value"],
            "intersection_size": term["intersection_size"],
            "term_size": term["term_size"],
        })
    results.sort(key=lambda x: x["p_value"])
    return results


# ===========================================================================
# Gene filtering
# ===========================================================================

def relative_abundance_to_clr(X, pseudo_count=1e-8):
    """Centered log-ratio (CLR) transform for compositional data.

    matrices.pkl stores RELATIVE ABUNDANCE (each fish's row sums to 1) —
    compositional data.  WGCNA needs log-scale expression, but the old
    pseudo-CPM transform (log2(RA*1e6+1)) did NOT remove the closure-induced
    structure: RA*1e6 is a constant total per fish (discarding library-depth
    variation), and a per-gene monotone log cannot undo the spurious negative
    correlations the unit-sum constraint forces across genes.

    CLR is the standard treatment for compositionality: log(x) - mean(log(x))
    per sample.  Zeros get a multiplicative pseudo-count replacement.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_genes)
        Non-negative relative abundances (rows sum to 1).
    pseudo_count : float
        Floor applied before log (zero-replacement).

    Returns
    -------
    np.ndarray (n_samples, n_genes) — CLR-transformed (not row-sum constrained).
    """
    X = np.asarray(X, dtype=np.float64)
    X = np.clip(X, pseudo_count, None)
    logX = np.log(X)
    return logX - logX.mean(axis=1, keepdims=True)


def filter_genes_two_stage(matrices, gene_names, min_expression=0.0,
                           top_n=None):
    """Two-stage gene filter: expression floor, then variance top-N.

    Returns (filtered_matrices, filtered_gene_names).
    """
    n_original = len(gene_names)
    if min_expression <= 0 and top_n is None:
        return matrices, gene_names
    if top_n is not None and top_n >= n_original:
        top_n = None

    # Pool expression for per-gene statistics.  Variance for the top-N filter
    # must be computed on the SAME scale WGCNA analyzes (CLR / log-scale).
    # Ranking on pooled raw relative abundance selected a different gene set —
    # raw-RA variance is dominated by a few high-abundance genes.
    all_expr = []
    for M in matrices.values():
        M = np.asarray(M, dtype=np.float64)
        if M.shape[0] > 0:
            all_expr.append(M)
    pooled = np.vstack(all_expr)
    # Expression floor is on RAW relative abundance (the same semantics the
    # CLI documents); variance ranking is on CLR (the analysis scale).
    raw_means = np.mean(pooled, axis=0)
    pooled_clr = relative_abundance_to_clr(pooled)
    gene_vars = np.var(pooled_clr, axis=0)
    keep_mask = np.ones(n_original, dtype=bool)

    # Stage 1: expression floor (raw relative-abundance mean)
    if min_expression > 0:
        expr_keep = raw_means >= min_expression
        n_removed = int((~expr_keep).sum())
        keep_mask &= expr_keep
        print(f"  Expression filter: removed {n_removed} genes "
              f"(mean < {min_expression:.2e}, {100*n_removed/n_original:.1f}%)")

    # Stage 2: variance top-N
    if top_n is not None:
        surviving_idx = np.where(keep_mask)[0]
        if top_n < len(surviving_idx):
            surv_vars = gene_vars[surviving_idx]
            var_order = np.argsort(surv_vars)[::-1]
            top_surviving = surviving_idx[var_order[:top_n]]
            new_mask = np.zeros(n_original, dtype=bool)
            new_mask[top_surviving] = True
            keep_mask = new_mask
            print(f"  Variance filter: kept top {top_n} genes "
                  f"(var range: {gene_vars[keep_mask].min():.2e} – "
                  f"{gene_vars[keep_mask].max():.2e})")

    final_idx = np.where(keep_mask)[0]
    final_idx.sort()
    print(f"  Final gene set: {len(final_idx)}/{n_original} "
          f"({100*len(final_idx)/n_original:.1f}%)")

    filtered_genes = [gene_names[i] for i in final_idx]
    filtered_matrices = {}
    for k, M in tqdm(matrices.items(), desc="  Subsetting matrices"):
        M = np.asarray(M, dtype=np.float64)
        filtered_matrices[k] = M[:, final_idx]

    return filtered_matrices, filtered_genes


# ===========================================================================
# Expression DataFrame construction
# ===========================================================================

def build_group_expression(matrices, keys, gene_names, sample_prefix):
    """Stack matrices into a samples × genes DataFrame.

    Returns (expr_df, sample_info_df).
    """
    if not keys:
        raise ValueError(f"No keys matched for prefix '{sample_prefix}'")

    arrays = []
    sample_ids = []
    for k in sorted(keys):
        M = np.asarray(matrices[k], dtype=np.float64)
        n_fish = M.shape[0]
        arrays.append(M)
        clean = str(k).replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_')
        sample_ids.extend([f'{sample_prefix}_{clean}_{i}' for i in range(n_fish)])

    X = np.vstack(arrays)
    # matrices.pkl stores RELATIVE ABUNDANCE (each fish's row sums to 1) —
    # compositional data.  The old pseudo-CPM transform (log2(RA*1e6+1)) did
    # NOT remove the closure-induced structure (RA*1e6 is a constant total per
    # fish, and the log is monotone, so the spurious cross-gene negative
    # correlations from the unit-sum constraint remained).  Use the centered
    # log-ratio transform — the standard treatment for compositional data.
    X = relative_abundance_to_clr(X)
    expr_df = pd.DataFrame(X, index=sample_ids, columns=gene_names)
    expr_df.index.name = 'sample_id'

    sample_info = pd.DataFrame({'condition': sample_prefix}, index=expr_df.index)
    sample_info.index.name = 'sample_id'

    print(f"  [{sample_prefix}] {expr_df.shape[0]} samples × "
          f"{expr_df.shape[1]} genes")
    return expr_df, sample_info


# ===========================================================================
# WGCNA runner
# ===========================================================================

def run_wgcna(name, expr_df, sample_info, output_dir):
    """Run PyWGCNA on one group. Returns WGCNA object."""
    from PyWGCNA.wgcna import WGCNA

    os.makedirs(output_dir, exist_ok=True)

    wgcna = WGCNA(
        name=name,
        species='stickleback',
        geneExp=expr_df,
        sampleInfo=sample_info,
        TPMcutoff=0,
        RsquaredCut=0.85,
        networkType='signed hybrid',
        TOMType='signed',
        minModuleSize=30,
        MEDissThres=0.1,
        save=True,
        outputPath=output_dir,
    )
    print(f"  [{name}] Running WGCNA on {expr_df.shape[0]} samples × "
          f"{expr_df.shape[1]} genes ...")
    try:
        wgcna.runWGCNA()
    except (Exception, SystemExit) as e:
        # PyWGCNA signals "all genes in one module" and other degenerate
        # cases via sys.exit() -> SystemExit (a BaseException), NOT a normal
        # Exception, so both must be caught.
        msg = (e.args[0] if (hasattr(e, 'args') and e.args) else str(e))
        print(f"  [{name}] WGCNA failed: {msg}")
        print(f"  [{name}] Skipping — too few samples or weak co-expression "
              f"in this group (WGCNA needs ~7+ samples for stable modules).")
        return None

    # LOUD soft-threshold quality check.  PyWGCNA silently falls back to the
    # highest-R2 power (typically the largest tested, e.g. 20) when NO power
    # reaches RsquaredCut.  On compositional-derived data the scale-free fit
    # often never qualifies, so the network may be built at an extreme power
    # without anyone noticing.  Surface it prominently.
    try:
        sft = getattr(wgcna, 'sft', None)
        if sft is not None:
            sft_r2 = getattr(sft, 'SFT.R.sq', None)
            sft_power = getattr(sft, 'powerEstimate', None) or getattr(
                sft, 'fitIndices', None)
            if sft_r2 is not None:
                best_r2 = float(np.max(sft_r2))
                if best_r2 < 0.85:
                    print(f"  [{name}] WARNING: no soft-threshold power reached "
                          f"RsquaredCut=0.85 (best scale-free R² = {best_r2:.3f}). "
                          f"PyWGCNA silently used the max-R² power; modules may "
                          f"be unreliable. Inspect the SFT curve before trusting "
                          f"these modules.")
    except Exception as exc:
        print(f"  [{name}] NOTE: could not inspect soft-threshold fit "
              f"({exc}) — skipping quality check.")

    modules = wgcna.getModuleName()
    # Count genes per module (exclude grey = unassigned)
    color_counts = (wgcna.datExpr.var['moduleColors']
                    .value_counts())
    non_grey = {c: n for c, n in color_counts.items() if c != 'grey'}
    print(f"  [{name}] {len(modules)} modules: "
          f"{sum(non_grey.values())} genes in {len(non_grey)} non-grey modules, "
          f"{color_counts.get('grey', 0)} unassigned")
    for color, count in sorted(non_grey.items(),
                                key=lambda x: -x[1])[:10]:
        print(f"    {color}: {count} genes")

    return wgcna


# ===========================================================================
# Differential expression check
# ===========================================================================

def differential_expression_check(expr_a, expr_b, label_a, label_b,
                                   alpha=0.05):
    """Mann-Whitney U per gene; report fraction DE."""
    from scipy.stats import mannwhitneyu

    n_genes = expr_a.shape[1]
    # Sample up to 2000 genes for speed
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_genes, min(2000, n_genes), replace=False)

    pvals = []
    for i in sample_idx:
        _, p = mannwhitneyu(expr_a.iloc[:, i], expr_b.iloc[:, i],
                            alternative='two-sided')
        pvals.append(p)

    pvals = np.array(pvals)
    n_sig_bonf = int(np.sum(pvals < alpha / len(sample_idx)))
    n_sig_nominal = int(np.sum(pvals < alpha))
    pct = 100 * n_sig_nominal / len(sample_idx)

    print(f"  DE check ({label_a} vs {label_b}): "
          f"{n_sig_nominal}/{len(sample_idx)} genes nominally DE ({pct:.1f}%), "
          f"{n_sig_bonf} Bonferroni-significant")
    if pct > 30:
        print(f"  NOTE: >30% DE — large between-group differences; "
              f"pooling would induce spurious correlations. "
              f"Separate-group WGCNA is the correct approach.")


# ===========================================================================
# Module preservation (both directions)
# ===========================================================================

def module_preservation_both_ways(wgcna_a, wgcna_b, label_a, label_b,
                                  output_dir):
    """Run compareNetworks with A as reference, then B as reference.

    Returns (pres_a_as_ref, pres_b_as_ref): each is a dict with keys
    'jaccard_df', 'fraction_df', 'pvalue_df', and the Comparison object.
    """
    from PyWGCNA.utils import compareNetworks

    os.makedirs(output_dir, exist_ok=True)

    # --- A as reference, B as test ---
    print(f"  Module preservation: {label_a} (ref) vs {label_b} (test) ...")
    comp_ab = None
    try:
        comp_ab = compareNetworks([wgcna_a, wgcna_b])
        comp_ab.compareNetworks()
        comp_ab.plotHeatmapComparison(
            color='coolwarm', row_cluster=True, col_cluster=True,
            save=True, plot_format='pdf',
            file_name=os.path.join(output_dir,
                                   f'{label_a}_ref_{label_b}_test_heatmap'))
    except BaseException as e:
        print(f"    Preservation/heatmap failed: {e}")

    # --- B as reference, A as test ---
    print(f"  Module preservation: {label_b} (ref) vs {label_a} (test) ...")
    comp_ba = None
    try:
        comp_ba = compareNetworks([wgcna_b, wgcna_a])
        comp_ba.compareNetworks()
        comp_ba.plotHeatmapComparison(
            color='coolwarm', row_cluster=True, col_cluster=True,
            save=True, plot_format='pdf',
            file_name=os.path.join(output_dir,
                                   f'{label_b}_ref_{label_a}_test_heatmap'))
    except BaseException as e:
        print(f"    Preservation/heatmap failed: {e}")

    def _safe_attrs(comp):
        if comp is None:
            return {'jaccard': None, 'fraction': None, 'pvalue': None, 'comp': None}
        return {'jaccard': comp.jaccard_similarity,
                'fraction': comp.fraction,
                'pvalue': comp.P_value,
                'comp': comp}

    return {
        'a_ref': _safe_attrs(comp_ab),
        'b_ref': _safe_attrs(comp_ba),
    }


def identify_preserved_modules(preservation, name_a, name_b,
                                jaccard_threshold=0.05,
                                pvalue_threshold=0.05):
    """Identify modules preserved in BOTH directions (cross-group only).

    compareNetworks prefixes module colours with the WGCNA object name,
    e.g. ``infected:black``.  We split on ``:`` to extract the original
    colour and only consider cross-group pairs (A module ↔ B module).

    Returns (preserved_colors, summary_df).
    """
    j_ab = preservation['a_ref']['jaccard']
    p_ab = preservation['a_ref']['pvalue']

    if j_ab is None or preservation['b_ref']['jaccard'] is None:
        print("  Module preservation failed — skipping preserved module "
              "identification")
        return set(), pd.DataFrame()

    # compareNetworks puts ALL modules on both axes, prefixed like
    #   "{wgcna_name}:{colour}".  Filter to cross-group pairs only.
    prefix_a = f'{name_a}:'
    prefix_b = f'{name_b}:'

    a_modules = [c for c in j_ab.index if c.startswith(prefix_a)]
    b_modules = [c for c in j_ab.columns if c.startswith(prefix_b)]

    preserved_pairs = set()
    rows = []

    for mod_a_full in a_modules:
        mod_a_color = mod_a_full.split(':', 1)[1]
        if mod_a_color == 'grey':
            continue
        best_b_full = None
        best_j = 0
        best_p = 1.0
        for mod_b_full in b_modules:
            mod_b_color = mod_b_full.split(':', 1)[1]
            if mod_b_color == 'grey':
                continue
            j = j_ab.loc[mod_a_full, mod_b_full]
            p = p_ab.loc[mod_a_full, mod_b_full]
            if j >= jaccard_threshold and p <= pvalue_threshold:
                if j > best_j:
                    best_j = j
                    best_p = p
                    best_b_full = mod_b_full
        if best_b_full is not None:
            # NOTE: the OLD code additionally required the reverse direction
            # (B-as-ref → A-as-test).  That check was VACUOUS: Jaccard is
            # symmetric and Fisher's exact p is invariant to transposing the
            # table, so j_ba[...] == j_ab[...] and p_ba[...] == p_ab[...]
            # exactly — the reverse branch could never reject a pair.  It has
            # been removed; a single-direction test is equivalent.
            mod_b_color = best_b_full.split(':', 1)[1]
            preserved_pairs.add(f'{mod_a_color}|{mod_b_color}')
            rows.append({
                'module_a': mod_a_color,
                'module_b': mod_b_color,
                'jaccard_ab': round(float(best_j), 4),
                'pvalue_ab': round(float(best_p), 6),
            })

    print(f"  Preserved modules: {len(preserved_pairs)} module PAIR(s) "
          f"(Jaccard ≥ {jaccard_threshold}, Fisher p ≤ {pvalue_threshold})")
    if preserved_pairs:
        print(f"    {' | '.join(sorted(preserved_pairs))}")

    # Return the set of preserved PAIRS — not the union of colors, which
    # would over-count when the two groups share one color palette.
    return preserved_pairs, pd.DataFrame(rows)


# ===========================================================================
# Eigengene → g:Profiler enrichment
# ===========================================================================

def eigengene_enrichment(wgcna_obj, gene_names, output_dir,
                          top_n_genes=100, resolver=None):
    """For each non-grey module: top kME genes → g:Profiler enrichment.

    Parameters
    ----------
    wgcna_obj : PyWGCNA.wgcna.WGCNA
    gene_names : list[str]
    output_dir : str
    top_n_genes : int
        Number of top kME genes to enrich per module.
    resolver : GeneNameResolver or None
        For LOC gene resolution.  If None, LOC genes are filtered out.

    Returns
    -------
    dict mapping module_color → list of enriched term dicts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get gene-to-module assignments
    module_colors_series = wgcna_obj.datExpr.var['moduleColors']
    expr_df = wgcna_obj.datExpr.to_df()  # samples × genes

    # Get eigengenes
    MEs = wgcna_obj.MEs  # samples × ME{color}

    enrichments = {}
    all_rows = []

    # Use only modules that actually appear in MEs
    me_colors = [c.replace('ME', '') for c in MEs.columns
                 if c.startswith('ME')]

    for color in tqdm(me_colors, desc="  Eigengene enrichment"):
        if color == 'grey':
            continue

        me_col = f'ME{color}'
        if me_col not in MEs.columns:
            continue

        me_values = MEs[me_col].values

        # Compute kME = |cor(gene_expression, ME)|
        # PyWGCNA may internally filter genes (low expression, outliers), so
        # ``datExpr`` can have fewer columns than ``gene_names``.  Use
        # name-based lookup to avoid IndexError on mismatched lengths.
        kme_scores = {}
        gene_module_bool = (module_colors_series == color)
        module_genes = set()
        # Map gene name → column position in expr_df
        expr_col_idx = {name: idx for idx, name in enumerate(expr_df.columns)}

        for gi, gene in enumerate(gene_names):
            if gene not in gene_module_bool.index:
                continue
            if not gene_module_bool[gene]:
                continue
            coli = expr_col_idx.get(gene)
            if coli is None:
                continue
            gene_expr = expr_df.iloc[:, coli].values.astype(np.float64)
            # Skip constant genes
            if np.std(gene_expr) == 0:
                continue
            corr = np.corrcoef(gene_expr, me_values)[0, 1]
            # SIGNED kME: the network is 'signed hybrid' (TOMType 'signed'),
            # so a module's hub genes are those POSITIVELY correlated with
            # the eigengene.  |corr| would rank anti-correlated genes (e.g.
            # r=-0.8) ahead of +0.75 and pull in genes that oppose the module.
            kme_scores[gi] = corr
            module_genes.add(gene_names[gi])

        if not kme_scores:
            continue

        # Top N by kME.  On a signed network only genes POSITIVELY correlated
        # with the eigengene are hubs; drop any negative-kME stragglers.
        pos_kme = {gi: k for gi, k in kme_scores.items() if k > 0}
        if not pos_kme:
            continue
        sorted_genes = sorted(pos_kme.items(), key=lambda x: -x[1])
        top_genes = [gene_names[gi] for gi, _ in sorted_genes[:top_n_genes]]

        # Filter to "named" genes using resolver (batch-resolve, then filter)
        if resolver is not None:
            resolved_names = resolver.resolve_many(top_genes, verbose=False)
        else:
            resolved_names = []
            for g in top_genes:
                clean = g.replace('.H', '')
                if (clean.lower().startswith('si.') or
                    clean.lower().startswith('trna') or
                    clean.upper().startswith('LOC')):
                    resolved_names.append(None)
                else:
                    resolved_names.append(clean)

        named = [r for r in resolved_names if r is not None]
        if len(named) < 5:
            continue

        # Convert → enrich
        ensembl_ids, _ = convert_to_ensembl(named)
        if len(ensembl_ids) < 5:
            tqdm.write(f"    {color}: only {len(ensembl_ids)} Ensembl IDs "
                       f"(need ≥5) — skipping enrichment")
            continue

        terms = run_enrichment(ensembl_ids)
        enrichments[color] = terms

        n_module = int(gene_module_bool.sum())
        if terms:
            tqdm.write(f"    {color}: {n_module} genes, top {len(named)} named, "
                       f"{len(ensembl_ids)} Ensembl, {len(terms)} enriched terms")
        else:
            tqdm.write(f"    {color}: {n_module} genes, {len(ensembl_ids)} Ensembl, "
                       f"0 enriched terms (none passed significance threshold)")

        # Collect rows for CSV
        for t in terms[:20]:  # top 20 terms per module
            all_rows.append({
                'module_color': color,
                'n_module_genes': n_module,
                'n_enriched': len(named),
                **t,
            })

    # Save combined CSV
    if all_rows:
        csv_path = os.path.join(output_dir, 'eigengene_enrichments.csv')
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"  Saved {len(all_rows)} enrichment rows to {csv_path}")

    return enrichments


# ===========================================================================
# VQ vs WGCNA comparison
# ===========================================================================

def compare_vq_vs_wgcna(matrices, gene_mappings, wgcna_objects,
                         comparison_label, output_dir, paradigm=None):
    """Compare VQ-code infection-associated genes vs WGCNA module genes.

    Parameters
    ----------
    matrices : dict
    gene_mappings : dict with vq_to_gene, gene_names
    wgcna_objects : dict mapping group_label → WGCNA object
    comparison_label : str (e.g. 'infection', 'sex')
    output_dir : str
    paradigm : str or None
        'sequential', 'joint', or None for backward-compat filename.
    """
    from scipy.stats import fisher_exact

    os.makedirs(output_dir, exist_ok=True)

    vq_to_gene = gene_mappings['vq_to_gene']
    gene_names_list = gene_mappings['gene_names']

    # --- Collect VQ code gene sets ---
    # For each VQ code, aggregate genes across all strata
    vq_gene_sets = {}
    for strat_key, code_to_genes in vq_to_gene.items():
        for code, gene_indices in code_to_genes.items():
            if code not in vq_gene_sets:
                vq_gene_sets[code] = set()
            for gi in gene_indices:
                if gi < len(gene_names_list):
                    vq_gene_sets[code].add(gene_names_list[gi])

    # --- Collect WGCNA module gene sets ---
    wgcna_gene_sets = {}
    for group_label, wgcna_obj in wgcna_objects.items():
        colors = wgcna_obj.datExpr.var['moduleColors']
        expr_genes = list(wgcna_obj.datExpr.var_names)
        for color in colors.unique():
            if color == 'grey':
                continue
            mask = (colors == color).values
            genes = set()
            for gi, is_in in enumerate(mask):
                if is_in and gi < len(expr_genes):
                    genes.add(expr_genes[gi])
            key = f"{group_label}/{color}"
            wgcna_gene_sets[key] = genes

    # --- Compute overlaps ---
    # Background: all genes in the WGCNA expression data (the filtered gene
    # set).  VQ codes can contain genes outside this set (from the full
    # gene_mappings), but those genes cannot appear in any WGCNA module.
    # Restrict VQ gene sets to the WGCNA background and use its size as the
    # Fisher universe so d counts all background genes absent from both sets.
    wgcna_background = set()
    for wgcna_obj in wgcna_objects.values():
        wgcna_background.update(wgcna_obj.datExpr.var_names)
    n_universe = len(wgcna_background)

    # Filter VQ gene sets to genes present in the WGCNA background
    vq_gene_sets = {code: genes & wgcna_background
                    for code, genes in vq_gene_sets.items()}

    rows = []
    for vq_code, vq_genes in tqdm(sorted(vq_gene_sets.items()),
                                   desc="  VQ-WGCNA overlap"):
        if len(vq_genes) < 5:
            continue
        for wgcna_key, wgcna_genes in sorted(wgcna_gene_sets.items()):
            if len(wgcna_genes) < 5:
                continue
            overlap = vq_genes & wgcna_genes
            if len(overlap) < 3:
                continue
            jaccard = len(overlap) / len(vq_genes | wgcna_genes)
            # Fisher exact
            a = len(overlap)
            b = len(vq_genes) - a
            c = len(wgcna_genes) - a
            d = n_universe - a - b - c
            _, fisher_p = fisher_exact([[a, b], [c, d]],
                                        alternative='greater')
            rows.append({
                'vq_code': vq_code,
                'wgcna_module': wgcna_key,
                'n_vq_genes': len(vq_genes),
                'n_wgcna_genes': len(wgcna_genes),
                'n_overlap': len(overlap),
                'jaccard': round(jaccard, 4),
                'fisher_p': round(float(fisher_p), 6),
            })

    if rows:
        df = pd.DataFrame(rows)
        # BH-FDR across ALL overlap tests in this comparison — dozens to
        # hundreds of tests with nominal p<0.05 would inflate the reported
        # significant count.
        df['fisher_q'] = benjamini_hochberg(df['fisher_p'].values)
        df.sort_values('jaccard', ascending=False, inplace=True)
        paradigm_suffix = f'_{paradigm}' if paradigm else ''
        csv_path = os.path.join(output_dir,
                                f'vq_wgcna_overlap_{comparison_label}{paradigm_suffix}.csv')
        df.to_csv(csv_path, index=False)
        n_sig = int((df['fisher_q'] < 0.05).sum())
        print(f"  VQ-WGCNA overlap: {len(df)} pairs, {n_sig} significant "
              f"(FDR q<0.05)")
        # Show top overlaps
        for _, row in df.head(10).iterrows():
            print(f"    VQ{row['vq_code']:3d} ↔ {row['wgcna_module']:30s}  "
                  f"J={row['jaccard']:.3f}  overlap={row['n_overlap']}  "
                  f"p={row['fisher_p']:.4f}")
        return df

    return None


# ===========================================================================
# VQ code enrichment → term-set comparison with WGCNA
# ===========================================================================

def run_vq_enrichment_from_csv(csv_path, gene_names, resolver,
                                output_dir, organism="gaculeatus",
                                delay=0.3):
    """Run g:Profiler on each VQ code's gene list from a *_genes_for_go.csv.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns: vq_code, p_value, ..., gene_indices
    gene_names : list[str]
    resolver : GeneNameResolver
    output_dir : str
    organism : str
    delay : float

    Returns
    -------
    dict : {vq_code: [{'term_id': ..., 'term_name': ..., 'p_value': ...}]}
    """
    if not os.path.exists(csv_path):
        print(f"  VQ enrichment: {csv_path} not found — skipping")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # The gene_indices / gene_names columns can be very large (all genes
    # assigned to a VQ code, semicolon-separated), exceeding Python's
    # default csv.field_size_limit of 131 KB.
    csv.field_size_limit(sys.maxsize)

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"  VQ enrichment: {len(rows)} significant codes from "
          f"{os.path.basename(csv_path)}")

    all_terms = {}
    for row in tqdm(rows, desc="  VQ g:Profiler", unit="code"):
        code = int(row['vq_code'])
        gene_idx_str = row.get('gene_indices', '')
        if not gene_idx_str:
            continue
        gene_indices = [int(g) for g in gene_idx_str.split(';') if g.strip()]
        raw_names = [gene_names[gi] for gi in gene_indices
                     if gi < len(gene_names)]
        if resolver is not None:
            resolved = resolver.resolve_many(raw_names, verbose=False)
        else:
            resolved = [g for g in raw_names
                        if not g.upper().startswith('LOC')
                        and not g.lower().startswith(('si.', 'si:', 'trna'))]
        named = [r for r in resolved if r is not None]
        if len(named) < 5:
            continue

        ensembl_ids, _ = convert_to_ensembl(named, organism)
        if len(ensembl_ids) < 5:
            continue

        terms = run_enrichment(ensembl_ids, organism)
        if terms:
            all_terms[code] = terms
        time.sleep(delay)

    # Save per-code enrichment
    if all_terms:
        out_rows = []
        for code, terms in all_terms.items():
            for t in terms:
                out_rows.append({
                    'vq_code': code,
                    **t,
                })
        out_path = os.path.join(output_dir, 'vq_code_enrichment.csv')
        df = pd.DataFrame(out_rows)
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(out_rows)} VQ enrichment rows to {out_path}")

    return all_terms


def compare_vq_wgcna_terms(vq_all_terms, wgcna_enrichment,
                             comparison_name, output_dir, paradigm=None):
    """Compare GO/KEGG term sets between VQ and WGCNA at stratification level.

    Parameters
    ----------
    vq_all_terms : dict {code: [term dicts]}
    wgcna_enrichment : dict {module_color: [term dicts]}
    comparison_name : str
    output_dir : str
    paradigm : str or None
        'sequential', 'joint', or None for backward-compat filename.
    """
    # Collect all unique terms from VQ
    vq_terms = {}    # term_id → {'term_name', 'best_p', 'n_codes'}
    for code, terms in vq_all_terms.items():
        for t in terms:
            tid = t['term_id']
            if tid not in vq_terms or t['p_value'] < vq_terms[tid]['best_p']:
                vq_terms[tid] = {
                    'term_name': t['term_name'],
                    'source': t['source'],
                    'best_p': t['p_value'],
                    'n_codes': 1,
                }
            else:
                vq_terms[tid]['n_codes'] += 1

    # Collect all unique terms from WGCNA
    wgcna_terms = {}  # term_id → {'term_name', 'best_p', 'n_modules'}
    for color, terms in wgcna_enrichment.items():
        for t in terms:
            tid = t['term_id']
            if tid not in wgcna_terms or t['p_value'] < wgcna_terms[tid]['best_p']:
                wgcna_terms[tid] = {
                    'term_name': t['term_name'],
                    'source': t['source'],
                    'best_p': t['p_value'],
                    'n_modules': 1,
                }
            else:
                wgcna_terms[tid]['n_modules'] += 1

    all_term_ids = set(vq_terms.keys()) | set(wgcna_terms.keys())
    vq_only = set(vq_terms.keys()) - set(wgcna_terms.keys())
    wgcna_only = set(wgcna_terms.keys()) - set(vq_terms.keys())
    shared = set(vq_terms.keys()) & set(wgcna_terms.keys())

    print(f"\n  === {comparison_name}: VQ vs WGCNA enrichment overlap ===")
    print(f"  VQ enriched terms:     {len(vq_terms)}")
    print(f"  WGCNA enriched terms:  {len(wgcna_terms)}")
    print(f"  Shared:                {len(shared)}")
    print(f"  VQ-only:               {len(vq_only)}")
    print(f"  WGCNA-only:            {len(wgcna_only)}")

    # Build comparison table
    rows = []
    for tid in sorted(all_term_ids):
        in_vq = tid in vq_terms
        in_wg = tid in wgcna_terms
        vq_info = vq_terms.get(tid, {})
        wg_info = wgcna_terms.get(tid, {})
        rows.append({
            'term_id': tid,
            'term_name': vq_info.get('term_name') or wg_info.get('term_name', ''),
            'source': vq_info.get('source') or wg_info.get('source', ''),
            'in_vq': in_vq,
            'in_wgcna': in_wg,
            'overlap': 'shared' if (in_vq and in_wg) else ('vq_only' if in_vq else 'wgcna_only'),
            'vq_best_p': vq_info.get('best_p'),
            'vq_n_codes': vq_info.get('n_codes', 0),
            'wgcna_best_p': wg_info.get('best_p'),
            'wgcna_n_modules': wg_info.get('n_modules', 0),
        })

    df = pd.DataFrame(rows)
    df.sort_values(['overlap', 'vq_best_p'], inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    paradigm_suffix = f'_{paradigm}' if paradigm else ''
    out_path = os.path.join(output_dir,
                            f'vq_wgcna_term_overlap_{comparison_name}{paradigm_suffix}.csv')
    df.to_csv(out_path, index=False)

    # Print top shared terms
    shared_df = df[df['overlap'] == 'shared'].head(15)
    if len(shared_df) > 0:
        print(f"\n  Top shared terms:")
        for _, row in shared_df.iterrows():
            print(f"    [{row['source']}] {row['term_name']}  "
                  f"VQ p={row['vq_best_p']:.2e}  "
                  f"WGCNA p={row['wgcna_best_p']:.2e}")

    print(f"  Saved term comparison to {out_path}")
    return df


# ===========================================================================
# Stratification comparison orchestrator
# ===========================================================================

def run_stratification_comparison(comparison_name, matrices, gene_names,
                                   gene_mappings, group_a_keys, group_b_keys,
                                   label_a, label_b, output_dir,
                                   min_expression, top_n_genes,
                                   resolver=None, vq_csv_paths=None,
                                   paradigm_gms=None,
                                   de_matrices=None, de_gene_names=None):
    """Run the full WGCNA pipeline for one comparison.

    Steps:
      1. Build expression DataFrames
      2. Differential expression check
      3. Run WGCNA on each group
      4. Two-way module preservation
      5. Identify preserved (consensus) modules
      6. Eigengene enrichment (g:Profiler)
      7. VQ vs WGCNA gene-set overlap (per paradigm when paradigm_gms given)
      8. VQ vs WGCNA enrichment term comparison
         (one pass per (path, paradigm) tuple in ``vq_csv_paths``)

    ``de_matrices`` / ``de_gene_names`` (optional): the FULL unfiltered
    expression data used for the differential-expression pre-check.  When
    omitted, the already-filtered ``matrices``/``gene_names`` are used, which
    made the DE check circular (variance-filtered genes are essentially
    guaranteed to be >30% DE).  Pass the full data to check the real
    between-group differences.

    Parameters
    ----------
    paradigm_gms : dict or None
        Optional mapping {paradigm: gene_mappings_dict}.  When provided and
        ``vq_csv_paths`` contains entries for multiple paradigms, step 7
        (gene-set overlap) runs once per paradigm using the correct mappings.
    """
    print(f"\n{'='*70}")
    print(f"  {comparison_name}: {label_a} vs {label_b}")
    print(f"{'='*70}")

    n_a = sum(matrices[k].shape[0] for k in group_a_keys)
    n_b = sum(matrices[k].shape[0] for k in group_b_keys)
    print(f"  {label_a}: {len(group_a_keys)} strata, {n_a} fish")
    print(f"  {label_b}: {len(group_b_keys)} strata, {n_b} fish")

    if n_a < 8 or n_b < 8:
        print(f"  SKIPPED: fewer than 8 fish per group "
              f"(requires ≥8 for WGCNA)")
        return None

    # 1. Build expression DataFrames
    expr_a, sinfo_a = build_group_expression(matrices, group_a_keys,
                                              gene_names, label_a)
    expr_b, sinfo_b = build_group_expression(matrices, group_b_keys,
                                              gene_names, label_b)

    # 2. Differential expression check — on the FULL (unfiltered) gene set when
    # provided, so the ">30% DE" conclusion isn't guaranteed by variance
    # filtering (it was circular on the top-N already-filtered genes).
    de_a, de_b = expr_a, expr_b
    if de_matrices is not None and de_gene_names is not None:
        de_a, _ = build_group_expression(de_matrices, group_a_keys,
                                         de_gene_names, label_a)
        de_b, _ = build_group_expression(de_matrices, group_b_keys,
                                         de_gene_names, label_b)
    differential_expression_check(de_a, de_b, label_a, label_b)

    # 3. Run WGCNA
    wgcna_dir_a = os.path.join(output_dir, f'{label_a}')
    wgcna_dir_b = os.path.join(output_dir, f'{label_b}')
    wgcna_a = run_wgcna(label_a, expr_a, sinfo_a, wgcna_dir_a)
    wgcna_b = run_wgcna(label_b, expr_b, sinfo_b, wgcna_dir_b)
    if wgcna_a is None or wgcna_b is None:
        print(f"  [{comparison_name}] Skipping — WGCNA failed for one or both "
              f"groups (too few samples or weak co-expression)")
        return None

    # 4. Two-way module preservation
    pres_dir = os.path.join(output_dir, f'{comparison_name}_preservation')
    preservation = module_preservation_both_ways(
        wgcna_a, wgcna_b, label_a, label_b, pres_dir)

    # 5. Identify preserved (consensus) modules
    preserved_colors, preserved_df = identify_preserved_modules(
        preservation, label_a, label_b)
    if len(preserved_df) > 0:
        csv_path = os.path.join(pres_dir, 'preserved_modules.csv')
        preserved_df.to_csv(csv_path, index=False)
        print(f"  Saved preserved modules to {csv_path}")

    # 6. Eigengene enrichment
    enrich_dir = os.path.join(output_dir, 'eigengene_enrichments')
    enrich_a = eigengene_enrichment(wgcna_a, gene_names,
                                     os.path.join(enrich_dir, label_a),
                                     resolver=resolver)
    enrich_b = eigengene_enrichment(wgcna_b, gene_names,
                                     os.path.join(enrich_dir, label_b),
                                     resolver=resolver)

    # 7 + 8. VQ vs WGCNA gene-set overlap + enrichment term comparison
    #         — one pass per paradigm so both steps use the correct gene
    #         mappings (step 7) and produce paradigm-labelled outputs.
    n_vq_enriched = 0
    n_shared_terms = 0
    if vq_csv_paths:
        combined_wgcna = {}
        combined_wgcna.update(enrich_a)
        combined_wgcna.update(enrich_b)
        for csv_path, paradigm in vq_csv_paths:
            # Step 7: gene-level overlap (uses paradigm-specific mappings if
            # available, so joint codes don't get resolved against sequential
            # gene→code assignments or vice versa).
            gm_paradigm = (paradigm_gms.get(paradigm) if paradigm_gms
                           else gene_mappings)
            comp_dir = os.path.join(output_dir, 'vq_wgcna_comparison')
            compare_vq_vs_wgcna(
                matrices, gm_paradigm,
                {label_a: wgcna_a, label_b: wgcna_b},
                comparison_name, comp_dir, paradigm=paradigm)

            # Step 8: VQ code GO enrichment
            term_dir = os.path.join(output_dir, 'vq_wgcna_term_comparison')
            vq_terms = run_vq_enrichment_from_csv(
                csv_path, gene_names, resolver, term_dir)
            if vq_terms:
                n_vq_enriched += len(vq_terms)
                term_df = compare_vq_wgcna_terms(
                    vq_terms, combined_wgcna, comparison_name, term_dir,
                    paradigm=paradigm)
                n_shared_terms += int(
                    (term_df['overlap'] == 'shared').sum())
            # Rename immediately so the next paradigm's pass doesn't
            # overwrite.  _write_enrichment_summary scans for
            # vq_code_enrichment*.csv and derives paradigm from suffix.
            para_path = os.path.join(term_dir,
                                     f'vq_code_enrichment_{paradigm}.csv')
            if os.path.exists(para_path):
                os.remove(para_path)
            default_path = os.path.join(term_dir, 'vq_code_enrichment.csv')
            if os.path.exists(default_path):
                os.rename(default_path, para_path)

    return {
        'comparison': comparison_name,
        'label_a': label_a, 'label_b': label_b,
        'n_a': n_a, 'n_b': n_b,
        'modules_a': wgcna_a.getModuleName(),
        'modules_b': wgcna_b.getModuleName(),
        'n_preserved': len(preserved_colors),
        'n_enriched_terms_a': sum(len(v) for v in enrich_a.values()),
        'n_enriched_terms_b': sum(len(v) for v in enrich_b.values()),
        'n_vq_enriched_terms': n_vq_enriched,
        'n_shared_terms': n_shared_terms,
    }


# ===========================================================================
# Main
# ===========================================================================

def resolve_vq_csvs(vq_figures_dir, filename):
    """Find ALL *_genes_for_go.csv files for a comparison across paradigms.

    Returns a list of (path, paradigm) tuples.  Paradigm is 'sequential' for
    ``figures/``, 'joint' for ``figures_joint/``.  Both may exist; zero, one,
    or two tuples are returned.  The caller should process every tuple so
    enrichment output reflects all available paradigms and codes are properly
    labelled.
    """
    if not vq_figures_dir:
        return []
    parent = os.path.dirname(vq_figures_dir.rstrip('/'))
    results = []
    # Always check BOTH dirs — never fall back silently.
    for dirname, paradigm in [('figures', 'sequential'),
                               ('figures_joint', 'joint')]:
        cand = os.path.join(parent, dirname, filename)
        if os.path.exists(cand):
            results.append((cand, paradigm))
    if not results:
        return []
    return results


def resolve_vq_csv(vq_figures_dir, filename):
    """Legacy: return a single path (preferring the given dir, falling back).

    Prefer ``resolve_vq_csvs`` for new code so both paradigms are captured.
    """
    if not vq_figures_dir:
        return None
    primary = os.path.join(vq_figures_dir, filename)
    if os.path.exists(primary):
        return primary
    base = os.path.basename(vq_figures_dir.rstrip('/'))
    parent = os.path.dirname(vq_figures_dir.rstrip('/'))
    sibling = 'figures' if base == 'figures_joint' else 'figures_joint'
    alt = os.path.join(parent, sibling, filename)
    if os.path.exists(alt):
        print(f"  VQ GO CSV: {filename} not in {base}/ — using {sibling}/ fallback")
        return alt
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive WGCNA with VQ comparison')

    # Data
    parser.add_argument('--matrices', required=True,
                        help='Path to matrices.pkl')
    parser.add_argument('--gene-mappings', required=True,
                        help='Path to gene_mappings.pkl (for gene names + VQ data)')

    # Output
    parser.add_argument('--output-dir', default='rol/output/wgcna',
                        help='Root output directory')

    # Gene filtering
    parser.add_argument('--top-n-genes', type=int, default=5000,
                        help='Keep top N genes by variance (default 5000; '
                             'set to 0 for all)')
    parser.add_argument('--min-expression', type=float, default=1e-5,
                        help='Remove genes with mean expression below this '
                             'threshold (default 1e-5)')

    # Comparison selection
    parser.add_argument('--year', type=int, default=2021,
                        help='Year for infection and sex comparisons '
                             '(default 2021)')
    parser.add_argument('--skip-infection', action='store_true')
    parser.add_argument('--skip-sex', action='store_true')
    parser.add_argument('--skip-year', action='store_true')
    parser.add_argument('--vq-figures-dir', default=None,
                        help='Directory containing *_genes_for_go.csv files '
                             'from pipeline.py step 9 for term comparison')

    # LOC gene resolution
    parser.add_argument('--loc-tsv', default='rol/output/gene_name_to_locid.tsv',
                        help='Pre-computed gene name → LOC ID TSV')
    parser.add_argument('--loc-cache', default='rol/output/ncbi_loc_cache.json',
                        help='NCBI LOC cache (fallback)')

    # WGCNA params
    parser.add_argument('--kme-top-n', type=int, default=100,
                        help='Top N kME genes per module for enrichment')
    parser.add_argument('--preservation-jaccard', type=float, default=0.05,
                        help='Jaccard threshold for module preservation')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    print("Loading data ...")
    with open(args.matrices, 'rb') as f:
        matrices = pickle.load(f)
    matrices = {k: v for k, v in matrices.items()
                if not str(k).startswith('__')}
    print(f"  {len(matrices)} matrices")

    with open(args.gene_mappings, 'rb') as f:
        gm = pickle.load(f)
    gene_names_full = list(gm['gene_names'])
    print(f"  {len(gene_names_full)} genes")

    # Auto-detect the other paradigm's gene_mappings file so step 7
    # (gene-set overlap) can use the correct code→gene assignments per
    # paradigm.  Both files share the same gene_names list; only
    # vq_to_gene / gene_to_vq differ.
    mappings_dir = os.path.dirname(os.path.abspath(args.gene_mappings))
    mappings_basename = os.path.basename(args.gene_mappings)
    if 'joint' in mappings_basename:
        this_paradigm = 'joint'
        other_paradigm = 'sequential'
        other_mappings_path = os.path.join(mappings_dir, 'gene_mappings.pkl')
    else:
        this_paradigm = 'sequential'
        other_paradigm = 'joint'
        other_mappings_path = os.path.join(mappings_dir, 'gene_mappings_joint.pkl')
    paradigm_gms = {this_paradigm: gm}
    if os.path.exists(other_mappings_path):
        with open(other_mappings_path, 'rb') as f:
            gm_other = pickle.load(f)
        paradigm_gms[other_paradigm] = gm_other
        print(f"  Loaded {other_paradigm} gene_mappings "
              f"({os.path.basename(other_mappings_path)})")
    else:
        print(f"  Note: {os.path.basename(other_mappings_path)} not found — "
              f"step 7 gene overlap will only cover {this_paradigm} paradigm")

    # --- Gene filtering ---
    # Keep the FULL unfiltered data for the differential-expression pre-check,
    # which must not run on the already variance-filtered genes (circular).
    matrices_full = matrices
    top_n = args.top_n_genes if args.top_n_genes > 0 else None
    min_expr = args.min_expression if args.min_expression > 0 else 0.0
    matrices, gene_names = filter_genes_two_stage(
        matrices, gene_names_full, min_expression=min_expr, top_n=top_n)

    # Set up gene name resolver (TSV first, NCBI fallback)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from scripts.loc_resolver import GeneNameResolver
    resolver = GeneNameResolver(
        tsv_path=args.loc_tsv if os.path.exists(args.loc_tsv) else None,
        ncbi_cache_path=args.loc_cache,
    )
    if resolver._tsv is not None:
        print(f"Loaded TSV with {len(resolver._tsv)} gene mappings")

    results = {}

    # --- Comparison 1: Infection ---
    if not args.skip_infection:
        yr = args.year
        inf_keys = [k for k in matrices
                    if re.search(rf'\({yr}\.\d\)-1$', str(k))]
        ninf_keys = [k for k in matrices
                     if re.search(rf'\({yr}\.\d\)-0$', str(k))]

        vq_infection_csvs = resolve_vq_csvs(
            args.vq_figures_dir, 'infection_genes_for_go.csv')
        res = run_stratification_comparison(
            'infection', matrices, gene_names, gm,
            inf_keys, ninf_keys, 'infected', 'noninfected',
            os.path.join(args.output_dir, 'infection'),
            args.min_expression, top_n, resolver=resolver,
            vq_csv_paths=vq_infection_csvs,
            paradigm_gms=paradigm_gms,
            de_matrices=matrices_full, de_gene_names=gene_names_full)
        if res:
            results['infection'] = res

    # --- Comparison 2: Sex ---
    if not args.skip_sex:
        yr = args.year
        f_keys = [k for k in matrices
                  if re.search(rf'\({yr}\.\d\)-f$', str(k))]
        m_keys = [k for k in matrices
                  if re.search(rf'\({yr}\.\d\)-m$', str(k))]

        vq_sex_csvs = resolve_vq_csvs(
            args.vq_figures_dir, 'sex_genes_for_go.csv')
        res = run_stratification_comparison(
            'sex', matrices, gene_names, gm,
            f_keys, m_keys, 'female', 'male',
            os.path.join(args.output_dir, 'sex'),
            args.min_expression, top_n, resolver=resolver,
            vq_csv_paths=vq_sex_csvs,
            paradigm_gms=paradigm_gms,
            de_matrices=matrices_full, de_gene_names=gene_names_full)
        if res:
            results['sex'] = res

    # --- Comparison 3: Year (per-lake earliest vs latest) ---
    if not args.skip_year:
        yl_keys = [k for k in matrices
                   if (re.search(r'\(\d{4}\.\d\)$', str(k))
                       and ')-' not in str(k)
                       and not str(k).endswith('-m')
                       and not str(k).endswith('-f'))]

        lake_year_counts = {}
        for k in yl_keys:
            s = str(k)
            lake = s.split(' (')[0]
            yr_match = re.search(r'\((\d{4})\.\d\)', s)
            if yr_match:
                yr = int(yr_match.group(1))
                lake_year_counts.setdefault(lake, {})[yr] = matrices[k].shape[0]

        qualifying = []
        for lake, years in lake_year_counts.items():
            sorted_yrs = sorted(years)
            if len(sorted_yrs) >= 2:
                early, late = sorted_yrs[0], sorted_yrs[-1]
                # 7 = minimum samples for a stable correlation estimate
                # (matches the pipeline min_fish; could become a CLI flag)
                if years[early] >= 7 and years[late] >= 7:
                    qualifying.append((lake, early, late,
                                       years[early] + years[late]))

        if not qualifying:
            print("\n  Year comparison SKIPPED: no lake has enough fish "
                  "at multiple timepoints")
        else:
            qualifying.sort(key=lambda x: -x[3])  # most fish first
            print(f"\n  Year contrast: {len(qualifying)} qualifying lakes "
                  f"(≥7 fish at both timepoints):")
            for lake, early, late, total in qualifying:
                print(f"    {lake}: {early}→{late} ({total} fish)")

            for lake, early, late, _ in qualifying:
                keys_early = [k for k in yl_keys
                              if str(k).startswith(lake)
                              and re.search(rf'\({early}\.\d\)$', str(k))]
                keys_late = [k for k in yl_keys
                             if str(k).startswith(lake)
                             and re.search(rf'\({late}\.\d\)$', str(k))]
                comparison = f'year_{lake}'

                res = run_stratification_comparison(
                    comparison, matrices, gene_names, gm,
                    keys_early, keys_late,
                    f'{lake}_{early}', f'{lake}_{late}',
                    os.path.join(args.output_dir, 'year', lake),
                    args.min_expression, top_n, resolver=resolver,
                    paradigm_gms=paradigm_gms,
                    de_matrices=matrices_full,
                    de_gene_names=gene_names_full)
                if res:
                    results[comparison] = res

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  WGCNA Summary")
    print(f"{'='*70}")
    for comp_name, r in results.items():
        print(f"  {comp_name}: {r['label_a']} ({r['n_a']} fish, "
              f"{len(r['modules_a'])} modules) vs "
              f"{r['label_b']} ({r['n_b']} fish, "
              f"{len(r['modules_b'])} modules) — "
              f"{r['n_preserved']} preserved modules")

    summary_path = os.path.join(args.output_dir, 'wgcna_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")


if __name__ == '__main__':
    main()
