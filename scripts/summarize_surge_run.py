#!/usr/bin/env python3
"""Summarize a SURGE run's manuscript stats, WGCNA results, and VQ↔WGCNA overlap.

Reads a local copy of a Modal volume output dir (see pull_modal_output.py) and
prints a human-readable report answering:

  1. What does the manuscript JSON report?
  2. What is the overlap between VQ (codebook) and WGCNA?
  3. What are the enriched genes/terms, if any?

It also recomputes the VQ-side code enrichment (infection/sex/role) from the
persisted embeddings, so the report can explain WHY the persisted overlap
files are empty (no VQ code reaches significance → no *_genes_for_go.csv was
exported → the WGCNA overlap steps had no VQ input).

Usage:
    python scripts/summarize_surge_run.py --data-dir output/10k_v1.0
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def benjamini_hochberg(pvals):
    """BH-FDR q-values; preserves input order. Mirrors scripts/wgcna_pipeline.py."""
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(pvals)
    qvals = np.empty(n)
    qvals[order] = np.minimum.accumulate(
        ranked * n / (np.arange(1, n + 1)))
    qvals = np.minimum(qvals, 1.0)
    # enforce monotonicity from the smallest p upward
    sorted_q = qvals[order]
    for i in range(n - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    qvals[order] = sorted_q
    return qvals


def code_enrichment(embeddings, key_filter, codebook_size, label):
    """Mann-Whitney per code between two key groups; returns code→{p,q,...}.

    key_filter(k) -> 0/1/None  (which group the key belongs to, or skip).
    Mirrors surge/analysis.py infection/sex/role_code_enrichment.
    """
    a_hist, b_hist = [], []
    for k in embeddings:
        grp = key_filter(k)
        if grp is None:
            continue
        (a_hist if grp == 1 else b_hist).append(embeddings[k])
    if not a_hist or not b_hist:
        print(f"  [{label}] no keys for both groups — skipping")
        return None
    a = np.array(a_hist)
    b = np.array(b_hist)
    from scipy.stats import mannwhitneyu
    results = {}
    for code in range(codebook_size):
        au = a[:, code]
        bu = b[:, code]
        pooled = np.concatenate([au, bu])
        if np.all(pooled == pooled[0]):
            continue  # identical across all strata — Mann-Whitney undefined
        u, p = mannwhitneyu(au, bu, alternative='two-sided')
        results[code] = {
            'a_mean': float(np.mean(au)),
            'b_mean': float(np.mean(bu)),
            'fold_change': (float(np.mean(au)) + 1e-8) / (float(np.mean(bu)) + 1e-8),
            'p_value': float(p),
        }
    codes = sorted(results.keys())
    qvals = benjamini_hochberg([results[c]['p_value'] for c in codes])
    for c, q in zip(codes, qvals):
        results[c]['q_value'] = float(q)
    return results


def _signif_counts(results):
    if not results:
        return 0, 0
    n_raw = sum(1 for r in results.values() if r['p_value'] < 0.05)
    n_q = sum(1 for r in results.values() if r['q_value'] < 0.05)
    return n_raw, n_q


def _fmt_p(p):
    if p is None:
        return '—'
    return f"{p:.1e}" if p < 1e-3 else f"{p:.4f}"


# ---------------------------------------------------------------------------
# WGCNA side
# ---------------------------------------------------------------------------

def load_wgcna_summary(wgcna_dir):
    path = os.path.join(wgcna_dir, 'wgcna_summary.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_eigengene_enrichments(wgcna_dir):
    """Return {comparison: {group: [rows]}} from wgcna/**/eigengene_enrichments/*/*.csv.

    Handles both layouts:
      wgcna/{comp}/eigengene_enrichments/{group}/...   (infection, sex)
      wgcna/year/{lake}/eigengene_enrichments/{group}/... (year contrasts)
    """
    import csv
    out = {}
    for csv_path in glob.glob(os.path.join(wgcna_dir, '**',
                                           'eigengene_enrichments.csv'),
                              recursive=True):
        rel = os.path.relpath(csv_path, wgcna_dir).split(os.sep)
        if rel[0] == 'year' and len(rel) >= 4:
            comp, group = f"year_{rel[1]}", rel[3]
        elif len(rel) >= 3:
            comp, group = rel[0], rel[2]
        else:
            continue
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        out.setdefault(comp, {})[group] = rows
    return out


# ---------------------------------------------------------------------------
# VQ side
# ---------------------------------------------------------------------------

def _inf_key(k):
    m = re.search(r'\)-([01])$', str(k))
    return None if not m else int(m.group(1))


def _sex_key(k):
    if str(k).endswith('-f'):
        return 1
    if str(k).endswith('-m'):
        return 0
    return None


def _role_key(k):
    # Source Benthic / Source Limnetic = Source (1); Recipient* = 0
    lake = str(k).split(' (')[0]
    return None  # role needs lake metadata (data.pkl); set by caller if available


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', '-d', default='output/10k_v1.0',
                    help='Local output dir pulled from the Modal volume')
    ap.add_argument('--codebook-size', '-k', type=int, default=100)
    args = ap.parse_args()

    dd = args.data_dir
    wgcna_dir = os.path.join(dd, 'wgcna')
    post = os.path.join(dd, 'postprocess')

    L = []
    def emit(line=''):
        print(line)
        L.append(line)

    emit("# SURGE run summary — %s" % os.path.basename(dd.rstrip('/')))
    emit()

    # ------------------------------------------------------------------
    # 1. Manuscript JSON
    # ------------------------------------------------------------------
    ms_path = os.path.join(post, 'manuscript_stats.json')
    if os.path.exists(ms_path):
        with open(ms_path) as f:
            ms = json.load(f)
        emit("## 1. Manuscript stats (manuscript_stats.json)")
        per = ms.get('permanova', {}).get('sequential', {})
        if per and isinstance(per, dict):
            for strat, rows in per.items():
                sig = [f"{k} (R²={v['r2']:.3f}, q={v['q_value']:.3f})"
                       for k, v in rows.items()
                       if isinstance(v, dict) and v.get('q_value', 1) < 0.05]
                emit(f"- PERMANOVA {strat}: significant → {sig if sig else 'none'}")
        ws = ms.get('wasserstein_and_slopes', {}).get('sequential', {})
        if ws and isinstance(ws, dict):
            emit(f"- Temporal drift: {ws.get('n_lakes')} lakes, "
                 f"mean slope={ws.get('mean_slope', float('nan')):.4f}, "
                 f"Wasserstein p95={ws.get('wasserstein_p95', float('nan')):.4f}")
            st = ws.get('stratifications', {}).get('year_lake', {})
            if isinstance(st, dict):
                emit(f"    year_lake: t-test p={st.get('ttest_pvalue')}, "
                     f"MW p={st.get('mannwhitney_pvalue')}, "
                     f"source_mean_slope={st.get('source_mean_slope')}, "
                     f"recipient_mean_slope={st.get('recipient_mean_slope')}")
        emit(f"- WGCNA enrichment present: {list(ms.get('wgcna_enrichment', {}).keys())}")
        emit(f"- VQ code enrichment present: {list(ms.get('vq_code_enrichment', {}).keys()) or 'EMPTY'}")
        emit(f"- VQ×WGCNA intersection present: {list(ms.get('vq_wgcna_intersection', {}).keys()) or 'EMPTY'}")
        emit()

    # ------------------------------------------------------------------
    # 2. VQ side: recompute code enrichment from embeddings
    # ------------------------------------------------------------------
    emb_path = os.path.join(dd, 'embeddings.pkl')
    if os.path.exists(emb_path):
        embeddings = load_pickle(emb_path)
        n_inf = sum(1 for k in embeddings if _inf_key(k) == 1)
        n_ninf = sum(1 for k in embeddings if _inf_key(k) == 0)
        n_f = sum(1 for k in embeddings if _sex_key(k) == 1)
        n_m = sum(1 for k in embeddings if _sex_key(k) == 0)

        emit("## 2. VQ code enrichment — recomputed from embeddings.pkl")
        emit(f"({len(embeddings)} strata, codebook size {args.codebook_size})")
        emit()

        inf = code_enrichment(embeddings, _inf_key, args.codebook_size, 'infection')
        nraw, nq = _signif_counts(inf)
        emit(f"### Infection  (infected n={n_inf}, noninfected n={n_ninf})")
        emit(f"  codes with raw p<0.05: {nraw}/{args.codebook_size}   "
             f"FDR q<0.05: {nq}/{args.codebook_size}")
        if inf:
            top = sorted(inf.values(), key=lambda r: r['q_value'])[:5]
            for r in top:
                emit(f"    code → p={_fmt_p(r['p_value'])} q={_fmt_p(r['q_value'])} "
                     f"FC={r['fold_change']:.2f}")
        emit()

        sex = code_enrichment(embeddings, _sex_key, args.codebook_size, 'sex')
        nraw, nq = _signif_counts(sex)
        emit(f"### Sex  (female n={n_f}, male n={n_m})")
        emit(f"  codes with raw p<0.05: {nraw}/{args.codebook_size}   "
             f"FDR q<0.05: {nq}/{args.codebook_size}")
        if sex:
            top = sorted(sex.values(), key=lambda r: r['q_value'])[:5]
            for r in top:
                emit(f"    code → p={_fmt_p(r['p_value'])} q={_fmt_p(r['q_value'])} "
                     f"FC={r['fold_change']:.2f}")
        emit()
    else:
        embeddings = None
        emit("## 2. VQ code enrichment — no embeddings.pkl found\n")

    # ------------------------------------------------------------------
    # 3. WGCNA side
    # ------------------------------------------------------------------
    wgs = load_wgcna_summary(wgcna_dir)
    if not wgs:
        emit("## 3. WGCNA — no wgcna_summary.json\n")
    else:
        emit("## 3. WGCNA summary (wgcna_summary.json)")
        comps = sorted(wgs.keys())
        year_lakes = [c for c in comps if c.startswith('year_')]
        for c in comps:
            info = wgs[c]
            if not isinstance(info, dict) or 'modules_a' not in info:
                continue
            emit(f"- **{c}**  {info.get('label_a')} (n={info.get('n_a')}) vs "
                 f"{info.get('label_b')} (n={info.get('n_b')})  "
                 f"[{len(info.get('modules_a', []))} vs {len(info.get('modules_b', []))} modules, "
                 f"{info.get('n_preserved')} preserved]")
        emit(f"\n  Year contrasts ({len(year_lakes)} lakes): "
             f"{', '.join(c.replace('year_', '') for c in year_lakes)}")
        # VQ/shared term counts
        vq_counts = {c: wgs[c].get('n_vq_enriched_terms')
                     for c in wgs if isinstance(wgs[c], dict)
                     and 'n_vq_enriched_terms' in wgs[c]}
        shared = {c: wgs[c].get('n_shared_terms')
                  for c in wgs if isinstance(wgs[c], dict)
                  and 'n_shared_terms' in wgs[c]}
        if vq_counts:
            emit(f"  VQ-enriched terms across comparisons: {set(vq_counts.values())}")
            emit(f"  Shared GO terms across comparisons: {set(shared.values())}")
        emit()

    # ------------------------------------------------------------------
    # 4. VQ ↔ WGCNA overlap — what is / isn't persisted
    # ------------------------------------------------------------------
    emit("## 4. VQ ↔ WGCNA overlap")
    overlap_files = glob.glob(os.path.join(wgcna_dir, '*', 'vq_wgcna_comparison',
                                           '*.csv'))
    vq_enrich_files = glob.glob(os.path.join(wgcna_dir, '*',
                                             'vq_wgcna_term_comparison',
                                             'vq_code_enrichment*.csv'))
    term_overlap_files = glob.glob(os.path.join(wgcna_dir, '*',
                                                'vq_wgcna_term_comparison',
                                                'vq_wgcna_term_overlap*.csv'))
    genes_for_go = glob.glob(os.path.join(dd, 'figures*', '*_genes_for_go.csv'))

    if overlap_files:
        emit(f"- Gene-level overlap CSVs found: {len(overlap_files)}")
        for f in overlap_files:
            emit(f"    {os.path.relpath(f, wgcna_dir)}")
    else:
        emit("- Gene-level overlap CSVs: **NONE** "
             "(vq_wgcna_comparison/ empty or absent)")
    emit(f"- VQ code enrichment CSVs: {len(vq_enrich_files) or 'NONE'}")
    emit(f"- Term-overlap CSVs: {len(term_overlap_files) or 'NONE'}")
    if genes_for_go:
        emit(f"- *_genes_for_go.csv exported: {len(genes_for_go)} → "
             f"{[os.path.basename(g) for g in genes_for_go]}")
    else:
        emit("- *_genes_for_go.csv: **NONE** (no VQ gene lists were exported)")
    emit()

    # Diagnosis
    emit("### Why is the overlap empty?")
    if embeddings is not None and inf is not None:
        _, nq_inf = _signif_counts(inf)
        if nq_inf == 0:
            emit("1. **VQ side: no significant codes.** Zero infection codes reach "
                 "FDR q<0.05 (and none at raw p<0.05 if `_signif_counts` shows 0). "
                 "The pipeline only exports *_genes_for_go.csv when codes pass, so "
                 "no VQ gene lists were produced for this run.")
        else:
            emit(f"1. VQ side has {nq_inf} significant infection codes but their "
                 "gene lists were not exported/persisted in this run.")
    emit("2. **WGCNA side: module→gene assignments were never persisted.** The "
         "per-group WGCNA dirs (wgcna/*/<group>/) are empty on the volume; only "
         "figure PDFs and eigengene *enrichment-term* CSVs survived. Without "
         "module gene lists, the gene-level overlap cannot be recomputed.")
    emit()

    # ------------------------------------------------------------------
    # 5. WGCNA eigengene enrichment terms (the 'enriched genes/terms')
    # ------------------------------------------------------------------
    enrich = load_eigengene_enrichments(wgcna_dir)
    emit("## 5. WGCNA eigengene GO enrichment (per comparison, top terms)")
    if not enrich:
        emit("  none found")
    else:
        for comp in sorted(enrich):
            groups = enrich[comp]
            emit(f"- **{comp}**")
            for group, rows in sorted(groups.items()):
                top = sorted(rows, key=lambda r: float(r.get('p_value', 1)))[:4]
                emit(f"    {group} ({len(rows)} terms): "
                     + "; ".join(f"{r.get('module_color')}·{r.get('term_name','')[:38]} "
                                 f"[{_fmt_p(float(r.get('p_value',1)))}]"
                                 for r in top))
        emit()

    # ------------------------------------------------------------------
    # 6. enrichment_summary.json status flags
    # ------------------------------------------------------------------
    es_path = os.path.join(post, 'enrichment_summary.json')
    if os.path.exists(es_path):
        with open(es_path) as f:
            es = json.load(f)
        flags = {k: v.get('status') for k, v in es.items()
                 if isinstance(v, dict) and 'status' in v}
        emit("## 6. enrichment_summary.json status flags")
        emit(f"  {flags}")
        emit()
        if flags.get('year') == 'failed':
            emit("  NOTE: `year` failed here because this summary was written "
                 "by pre-fix code that looked up the literal 'year' key while "
                 "wgcna_pipeline stores 'year_<Lake>'. The per-lake data "
                 "exists; a rerun with the fix reports each lake's year table.")
            emit()

    # ------------------------------------------------------------------
    # Save markdown alongside the data
    # ------------------------------------------------------------------
    out_path = os.path.join(post, 'surge_run_summary.md')
    try:
        with open(out_path, 'w') as f:
            f.write("\n".join(L) + "\n")
        print(f"\n[saved] {out_path}")
    except OSError as e:
        print(f"\n[!] could not write {out_path}: {e}")


if __name__ == '__main__':
    main()
