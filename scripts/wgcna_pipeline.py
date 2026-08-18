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
import subprocess
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


def _hypergeom_enrichment(in_cluster, cluster_size, bg_size, bg_hit):
    """Upper-tail hypergeometric enrichment p-value.

    P(X >= in_cluster) drawing ``cluster_size`` items from a population of
    ``bg_size`` containing ``bg_hit`` successes.  Conditioning only on the
    cluster size and the background trait count gives the fully-contained case
    the tiny probability it deserves, unlike Fisher's exact test which
    conditions on both margins and reports p = 1.0 there (mirrors
    surge/analysis.py).
    """
    from scipy.stats import hypergeom
    if in_cluster <= 0 or bg_hit <= 0 or cluster_size <= 0:
        return 1.0
    return float(hypergeom.sf(in_cluster - 1, bg_size, bg_hit, cluster_size))


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

class _DatExprShim:
    """Lightweight stand-in for PyWGCNA's ``datExpr`` attribute.

    Exposes only the members the downstream steps read: ``var['moduleColors']``
    (pandas Series gene→color), ``var_names`` (list), and ``to_df()`` (the
    samples × genes DataFrame, aligned to the genes WGCNA actually used).
    """

    def __init__(self, colors_series, expr_genes, df):
        self.var = {'moduleColors': colors_series}
        self.var_names = expr_genes
        self._df = df

    def to_df(self):
        return self._df


class _WGCNAResult:
    """Lightweight stand-in for a PyWGCNA WGCNA object, rebuilt from the
    pickle an isolated group run returns.  ``sft`` is kept for interface
    compatibility; the soft-threshold quality check already ran in the
    subprocess."""

    def __init__(self, datExpr, MEs, sft=None):
        self.datExpr = datExpr
        self.MEs = MEs
        self.sft = sft

    def getModuleName(self):
        """List of module colours (incl. grey), matching PyWGCNA.getModuleName."""
        return list(self.datExpr.var['moduleColors'].dropna().unique())


def _load_wgcna_result(name, expr_df, output_dir):
    """Rebuild a :class:`_WGCNAResult` from an isolated group run's pickle."""
    with open(os.path.join(output_dir, 'wgcna_result.pkl'), 'rb') as f:
        result = pickle.load(f)
    expr_genes = list(result['expr_genes'])
    colors = pd.Series(result['moduleColors'], index=expr_genes,
                       name='moduleColors')
    # Align the expression to BOTH the genes and the samples WGCNA actually
    # used (its pre-processing can drop outlier samples; kME correlates the
    # per-gene expression vector with the eigengene, so equal lengths matter).
    expr_w = expr_df.reindex(columns=expr_genes)
    me_df = result['MEs']
    if me_df is not None and len(me_df):
        expr_w = expr_w.reindex(index=me_df.index)
        if expr_w.isna().any().any():
            print(f"  [{name}] WARNING: expression/MEs sample alignment after "
                  f"WGCNA filtering has {int(expr_w.isna().sum().sum())} NaN — "
                  f"kME for affected genes will be skipped.")
    datExpr = _DatExprShim(colors, expr_genes, expr_w)
    return _WGCNAResult(datExpr, me_df)


def run_wgcna(name, expr_df, sample_info, output_dir, retries=1):
    """Run PyWGCNA on one group in an ISOLATED subprocess.

    PyWGCNA embeds R in the calling process, and running a SECOND large WGCNA
    in the same embedded R session intermittently SIGSEGVs in WGCNA's C-level
    connectivity code (``pickSoftThreshold``) — a native crash that cannot be
    caught in Python.  Running each group in its own subprocess gives every
    WGCNA a fresh R session (the first run in a fresh session always worked)
    and lets a crash be contained + retried instead of killing the pipeline.

    Returns a :class:`_WGCNAResult` shim exposing the attributes downstream
    steps need (``.datExpr.var['moduleColors']``, ``.datExpr.var_names``,
    ``.datExpr.to_df()``, ``.MEs``), or ``None`` if the group's WGCNA failed
    after ``retries`` attempts.
    """
    os.makedirs(output_dir, exist_ok=True)
    expr_pkl = os.path.join(output_dir, '_expr.pkl')
    sinfo_pkl = os.path.join(output_dir, '_sinfo.pkl')
    result_pkl = os.path.join(output_dir, 'wgcna_result.pkl')
    with open(expr_pkl, 'wb') as f:
        pickle.dump(expr_df, f)
    with open(sinfo_pkl, 'wb') as f:
        pickle.dump(sample_info, f)

    cmd = [sys.executable, os.path.abspath(__file__), '--single-group',
           name, expr_pkl, sinfo_pkl, output_dir]

    ok = False
    for attempt in range(retries + 1):
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            ok = True
            break
        if proc.returncode == 2:
            print(f"  [{name}] WGCNA skipped by subprocess "
                  f"(degenerate / too few samples)")
            break
        print(f"  [{name}] WGCNA crashed (rc={proc.returncode}"
              + (" SIGSEGV" if proc.returncode == -11 else "")
              + f") — attempt {attempt + 1}/{retries + 1}")
        if os.path.exists(result_pkl):
            os.remove(result_pkl)

    for p in (expr_pkl, sinfo_pkl):
        if os.path.exists(p):
            os.remove(p)

    if not ok:
        print(f"  [{name}] WGCNA FAILED after {retries + 1} attempt(s) — "
              f"group marked failed")
        return None
    return _load_wgcna_result(name, expr_df, output_dir)


def _run_wgcna_inner(name, expr_df, sample_info, output_dir):
    """Run PyWGCNA on one group (in the CURRENT process). Returns WGCNA object.

    This is the actual PyWGCNA execution.  It is invoked via subprocess
    (see :func:`run_wgcna`) so every group gets a fresh embedded R session.
    """
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
    """Module-overlap preservation statistics between two groups' WGCNA modules.

    Pure-Python replacement for PyWGCNA's ``compareNetworks``: that API needs
    the full R objects, which cannot be pickled out of the isolated WGCNA
    subprocesses, and the pairwise module-overlap statistics it computes
    (Jaccard similarity, overlap fraction, hypergeometric overlap p-value) are
    trivially reproduced from the two groups' module assignments.

    Returns the same dict shape the caller expects — ``{a_ref, b_ref}``, each
    with ``jaccard`` / ``fraction`` / ``pvalue`` DataFrames whose rows/columns
    are labelled ``{name}:{colour}``.  Jaccard and the hypergeometric p are
    symmetric under a↔b transposition, so ``b_ref`` carries the same values;
    the reverse direction is kept only for interface compatibility.  Heatmap
    PDFs (cosmetic) are replaced by CSV exports of the two matrices.
    """
    colors_a = wgcna_a.datExpr.var['moduleColors']
    colors_b = wgcna_b.datExpr.var['moduleColors']
    universe = set(colors_a.index) & set(colors_b.index)

    def module_sets(colors, genes):
        out = {}
        for g in genes:
            c = colors.get(g)
            if c is None or c == 'grey':
                continue
            out.setdefault(c, set()).add(g)
        return out

    mods_a = module_sets(colors_a, universe)
    mods_b = module_sets(colors_b, universe)
    n_universe = len(universe)

    def overlap_frame(ref_label, test_label, ref_mods, test_mods):
        rows = sorted(ref_mods)
        cols = sorted(test_mods)
        idx = [f'{ref_label}:{r}' for r in rows]
        col_labels = [f'{test_label}:{c}' for c in cols]
        jac = pd.DataFrame(0.0, index=idx, columns=col_labels)
        frac = pd.DataFrame(0.0, index=idx, columns=col_labels)
        pval = pd.DataFrame(1.0, index=idx, columns=col_labels)
        for ra in rows:
            a = ref_mods[ra]
            for cb in cols:
                b = test_mods[cb]
                o = len(a & b)
                jac.loc[f'{ref_label}:{ra}', f'{test_label}:{cb}'] = (
                    o / (len(a) + len(b) - o) if (len(a) + len(b) - o) else 0.0)
                frac.loc[f'{ref_label}:{ra}', f'{test_label}:{cb}'] = (
                    o / len(b) if b else 0.0)
                # upper-tail hypergeometric: P(overlap >= o | U, |a|, |b|)
                pval.loc[f'{ref_label}:{ra}', f'{test_label}:{cb}'] = (
                    _hypergeom_enrichment(o, len(b), n_universe, len(a)))
        return jac, frac, pval

    print(f"  Module preservation: {label_a} (ref) vs {label_b} (test) ...")
    j_ab, frac_ab, p_ab = overlap_frame(label_a, label_b, mods_a, mods_b)
    print(f"  Module preservation: {label_b} (ref) vs {label_a} (test) ...")
    j_ba, frac_ba, p_ba = overlap_frame(label_b, label_a, mods_b, mods_a)

    os.makedirs(output_dir, exist_ok=True)
    j_ab.to_csv(os.path.join(output_dir, 'module_jaccard.csv'))
    p_ab.to_csv(os.path.join(output_dir, 'module_pvalue.csv'))

    return {
        'a_ref': {'jaccard': j_ab, 'fraction': frac_ab, 'pvalue': p_ab,
                  'comp': None},
        'b_ref': {'jaccard': j_ba, 'fraction': frac_ba, 'pvalue': p_ba,
                  'comp': None},
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

    # Persist the module→gene assignments so the VQ↔WGCNA gene-level overlap
    # is reproducible after the run.  Previously only figure PDFs and
    # enrichment terms were kept, and the per-group WGCNA dirs were empty on
    # the volume — the module membership (datExpr.var['moduleColors']) lived
    # only in the ephemeral container and was unreconstructable downstream.
    module_genes_path = os.path.join(output_dir, 'module_genes.csv')
    n_non_grey = int((module_colors_series != 'grey').sum())
    with open(module_genes_path, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['gene', 'module_color'])
        for gene, color in module_colors_series.items():
            wcsv.writerow([gene, color])
    print(f"  Saved {len(module_colors_series)} gene-module assignments "
          f"({n_non_grey} non-grey) to {module_genes_path}")

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
            # Upper-tail hypergeometric p (NOT Fisher's exact test): Fisher
            # conditions on BOTH margins, so a fully-contained overlap (the
            # VQ code contains the whole module — the STRONGEST enrichment)
            # reports p=1.0 and no pair survives FDR.  Conditioning only on
            # the module size and the background code size gives the strongest
            # overlaps the tiny p they deserve.  Mirrors _hypergeom_enrichment
            # used by the eigengene/cluster enrichments.  Column kept as
            # 'fisher_p' for backward compatibility with consumers.
            p_overlap = _hypergeom_enrichment(
                len(overlap), len(wgcna_genes), n_universe, len(vq_genes))
            rows.append({
                'vq_code': vq_code,
                'wgcna_module': wgcna_key,
                'n_vq_genes': len(vq_genes),
                'n_wgcna_genes': len(wgcna_genes),
                'n_overlap': len(overlap),
                'jaccard': round(jaccard, 4),
                'fisher_p': round(float(p_overlap), 6),
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
                                   resolver=None, vq_csv_paths=None,
                                   paradigm_gms=None,
                                   de_matrices=None, de_gene_names=None,
                                   kme_top_n=100, jaccard_threshold=0.05):
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

    # Minimum matches the pipeline's min_fish=7 (a stratum is only built with
    # >=7 fish), so a group is not printed as qualifying and then skipped.
    if n_a < 7 or n_b < 7:
        print(f"  SKIPPED: fewer than 7 fish per group "
              f"(requires ≥7, matching the pipeline min_fish)")
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
        preservation, label_a, label_b,
        jaccard_threshold=jaccard_threshold)
    if len(preserved_df) > 0:
        csv_path = os.path.join(pres_dir, 'preserved_modules.csv')
        preserved_df.to_csv(csv_path, index=False)
        print(f"  Saved preserved modules to {csv_path}")

    # 6. Eigengene enrichment
    enrich_dir = os.path.join(output_dir, 'eigengene_enrichments')
    enrich_a = eigengene_enrichment(wgcna_a, gene_names,
                                     os.path.join(enrich_dir, label_a),
                                     top_n_genes=kme_top_n,
                                     resolver=resolver)
    enrich_b = eigengene_enrichment(wgcna_b, gene_names,
                                     os.path.join(enrich_dir, label_b),
                                     top_n_genes=kme_top_n,
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
            # The CSV gene_indices index into the FULL model gene array
            # (gene_mappings['gene_names'], the set the VQ model was trained
            # on), NOT the re-filtered WGCNA gene_names — using the shorter
            # array here silently resolved the wrong gene symbols (and dropped
            # any index >= its length), corrupting the g:Profiler queries.
            vq_terms = run_vq_enrichment_from_csv(
                csv_path, gene_mappings['gene_names'], resolver, term_dir)
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

    # Data.  Not required=True: the --single-group mode (spawned by run_wgcna)
    # needs neither — it reads an already-built expression DataFrame.
    parser.add_argument('--matrices',
                        help='Path to matrices.pkl (required for the main run)')
    parser.add_argument('--gene-mappings',
                        help='Path to gene_mappings.pkl (for gene names + VQ '
                             'data; required for the main run)')

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
    parser.add_argument('--single-group', nargs=4,
                        metavar=('NAME', 'EXPR_PKL', 'SINFO_PKL', 'OUT_DIR'),
                        help='Isolated single-group mode (invoked by run_wgcna '
                             'via subprocess): run WGCNA for ONE group in this '
                             'process with a fresh embedded R session and '
                             'pickle the essential results.')

    args = parser.parse_args()

    # --- Isolated single-group mode ---
    # Each group's WGCNA runs in its own process so every PyWGCNA call gets a
    # fresh embedded R session.  A second large WGCNA in one session SIGSEGVs
    # in WGCNA's C connectivity code (pickSoftThreshold); isolation makes every
    # group a "first run in a fresh session" (which always worked).
    if args.single_group is not None:
        name, expr_pkl, sinfo_pkl, out_dir = args.single_group
        with open(expr_pkl, 'rb') as f:
            expr_df = pickle.load(f)
        with open(sinfo_pkl, 'rb') as f:
            sample_info = pickle.load(f)
        wgcna = _run_wgcna_inner(name, expr_df, sample_info, out_dir)
        if wgcna is None:
            sys.exit(2)
        result = {
            'expr_genes': list(wgcna.datExpr.var_names),
            'moduleColors': list(wgcna.datExpr.var['moduleColors']),
            'MEs': wgcna.MEs.copy(),
        }
        with open(os.path.join(out_dir, 'wgcna_result.pkl'), 'wb') as f:
            pickle.dump(result, f)
        print(f"[single-group] {name}: WGCNA complete — "
              f"{len(result['expr_genes'])} genes")
        sys.exit(0)

    # Main (comparison) run: --matrices / --gene-mappings are mandatory.
    if not args.matrices or not args.gene_mappings:
        parser.error('--matrices and --gene-mappings are required for the '
                     'main run (only --single-group skips them)')

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

    # --- Gene set: INHERITED from the SURGE pipeline, not re-filtered ---
    # The SURGE pipeline already selected the gene set (gene_mappings
    # ['gene_names'] = the genes the VQ model was trained on) and saved
    # matrices.pkl ALIGNED to it (columns == gene_names order).  WGCNA must
    # run on EXACTLY that gene set for the SURGE-vs-WGCNA comparison to be
    # apples-to-apples — any WGCNA-side re-filtering (the old
    # filter_genes_two_stage, default top-5000) silently compared WGCNA on a
    # DIFFERENT gene set.  The --top-n-genes/--min-expression flags here are
    # therefore IGNORED.
    matrices_full = matrices
    gene_names = gene_names_full
    if args.top_n_genes != 0 or args.min_expression > 0:
        print("[wgcna] NOTE: --top-n-genes/--min-expression are ignored — the "
              "gene set is inherited from the SURGE pipeline "
              "(gene_mappings['gene_names'], "
              f"{len(gene_names)} genes). Pass 0/0.0 to silence this.")

    # Validate alignment: matrices columns must correspond 1:1 to gene_names.
    sample_M = next(iter(matrices.values()))
    if sample_M.shape[1] != len(gene_names):
        raise ValueError(
            f"[wgcna] matrices have {sample_M.shape[1]} columns but "
            f"gene_mappings has {len(gene_names)} genes. matrices.pkl must be "
            f"saved aligned to the model gene set — rerun the SURGE pipeline "
            f"with the same --top-n-genes (the pipeline now persists the "
            f"filtered matrices).")

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
        # Years may be int or float in the metadata ('(2021)' or '(2021.0)'),
        # so the year part must be OPTIONALLY decimal — a strict `\.\d`
        # matched nothing for integer-dtype Year columns and silently skipped
        # every comparison (empty keys → n_a=0 < min → no-op).
        inf_keys = [k for k in matrices
                    if re.search(rf'\({yr}(?:\.\d+)?\)-1$', str(k))]
        ninf_keys = [k for k in matrices
                     if re.search(rf'\({yr}(?:\.\d+)?\)-0$', str(k))]

        vq_infection_csvs = resolve_vq_csvs(
            args.vq_figures_dir, 'infection_genes_for_go.csv')
        res = run_stratification_comparison(
            'infection', matrices, gene_names, gm,
            inf_keys, ninf_keys, 'infected', 'noninfected',
            os.path.join(args.output_dir, 'infection'),
            resolver=resolver,
            vq_csv_paths=vq_infection_csvs,
            paradigm_gms=paradigm_gms,
            de_matrices=matrices_full, de_gene_names=gene_names_full,
            kme_top_n=args.kme_top_n,
            jaccard_threshold=args.preservation_jaccard)
        if res:
            results['infection'] = res

    # --- Comparison 2: Sex ---
    if not args.skip_sex:
        yr = args.year
        f_keys = [k for k in matrices
                  if re.search(rf'\({yr}(?:\.\d+)?\)-f$', str(k))]
        m_keys = [k for k in matrices
                  if re.search(rf'\({yr}(?:\.\d+)?\)-m$', str(k))]

        vq_sex_csvs = resolve_vq_csvs(
            args.vq_figures_dir, 'sex_genes_for_go.csv')
        res = run_stratification_comparison(
            'sex', matrices, gene_names, gm,
            f_keys, m_keys, 'female', 'male',
            os.path.join(args.output_dir, 'sex'),
            resolver=resolver,
            vq_csv_paths=vq_sex_csvs,
            paradigm_gms=paradigm_gms,
            de_matrices=matrices_full, de_gene_names=gene_names_full,
            kme_top_n=args.kme_top_n,
            jaccard_threshold=args.preservation_jaccard)
        if res:
            results['sex'] = res

    # --- Comparison 3: Year (per-lake earliest vs latest) ---
    if not args.skip_year:
        yl_keys = [k for k in matrices
                   if (re.search(r'\(\d{4}(?:\.\d+)?\)$', str(k))
                       and ')-' not in str(k)
                       and not str(k).endswith('-m')
                       and not str(k).endswith('-f'))]

        lake_year_counts = {}
        for k in yl_keys:
            s = str(k)
            lake = s.split(' (')[0]
            yr_match = re.search(r'\((\d{4})(?:\.\d+)?\)', s)
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
                    resolver=resolver,
                    paradigm_gms=paradigm_gms,
                    de_matrices=matrices_full,
                    de_gene_names=gene_names_full,
                    kme_top_n=args.kme_top_n,
                    jaccard_threshold=args.preservation_jaccard)
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
