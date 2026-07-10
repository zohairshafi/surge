"""
Batch correction for HK transcriptome data: log-CPM + ComBat.

Python equivalent of 00-batch_correction_expression_data.R

Applies log-CPM normalization (log2 counts-per-million) followed by ComBat batch
correction to remove technical effects from samples being processed by
different personnel across years (Lindsay: 2019-2022; Rogini: 2023).

Outputs per lake-type (source/recipient):
  - expression_batch_corrected.csv  (genes x samples)
  - design_with_batch.csv
  - batch_correction_validation.csv
"""

import os
import numpy as np
import pandas as pd
from pycombat import Combat


def _assign_batch(year_series):
    """Assign batch labels: Lindsay (2019–2022) vs Rogini (2023).

    Uses numeric comparison so float years (2019.0, 2020.0, …) are
    handled correctly — ``.astype(str)`` on a float would produce
    ``"2019.0"`` which fails a string match against ``"2019"``.
    """
    year_num = pd.to_numeric(year_series, errors='coerce')
    return np.where((year_num >= 2019) & (year_num <= 2022),
                    "Lindsay", "Rogini")


def _select_top_genes_and_track(expr_df, n_top=10000, method='expression'):
    """Select top n_top genes from a DataFrame, returning both the filtered
    DataFrame and the list of kept gene names."""
    if method == 'variance':
        gene_scores = expr_df.var(axis=0)
    else:
        gene_scores = expr_df.sum(axis=0)
    top_idx = gene_scores.argsort()[::-1][:n_top]
    kept_genes = list(expr_df.columns[top_idx])
    return expr_df.iloc[:, top_idx], kept_genes


def _select_top_genes(expr_matrix, n_top=10000, method='expression'):
    """Select top n_top genes by expression or variability across samples.

    Parameters
    ----------
    expr_matrix : pd.DataFrame or np.ndarray, shape (samples, genes)
    n_top : int
        Number of top genes to retain.
    method : str, 'expression' or 'variance'
        'expression': top by total expression (sum across samples).
        'variance': top by variance across samples.

    Returns
    -------
    expr_subset : same type as input, subset to top genes
    """
    if isinstance(expr_matrix, pd.DataFrame):
        if method == 'variance':
            gene_scores = expr_matrix.var(axis=0)
        else:
            gene_scores = expr_matrix.sum(axis=0)
        top_idx = gene_scores.argsort()[::-1][:n_top]
        return expr_matrix.iloc[:, top_idx]
    else:
        if method == 'variance':
            gene_scores = expr_matrix.var(axis=0)
        else:
            gene_scores = expr_matrix.sum(axis=0)
        top_idx = np.argsort(gene_scores)[::-1][:n_top]
        return expr_matrix[:, top_idx]


def run_combat(rna_data, design_data, output_dir,
               combat_formula=None, n_top=10000, filter_method='expression'):
    """Apply log-CPM + ComBat to a single cohort (source or recipient).

    Parameters
    ----------
    rna_data : pd.DataFrame, shape (samples, genes)
        Raw count expression matrix.
    design_data : pd.DataFrame
        Must have columns: Lake, Year, Batch (added in-place).
    output_dir : str
        Directory for output CSVs.
    combat_formula : str or None
        R-style formula for ComBat design matrix. E.g. "~ Lake".
        If None, automatically uses "~ Lake" when lakes exist in both
        batches, otherwise uses null design.
    n_top : int
        Number of top genes to retain for batch correction (default 10000).
    filter_method : str
        'expression' (sum) or 'variance' for selecting top genes.

    Returns
    -------
    dict with keys: expr_corrected (np.ndarray, genes x samples),
                    design (pd.DataFrame), validation (pd.DataFrame),
                    pc1_before, pc1_after, pc1_reduction
    """
    os.makedirs(output_dir, exist_ok=True)

    # Assign batch
    design_data = design_data.copy()
    design_data['Batch'] = _assign_batch(design_data['Year'])

    n_lindsay = (design_data['Batch'] == 'Lindsay').sum()
    n_rogini = (design_data['Batch'] == 'Rogini').sum()
    print(f"  Batch: Lindsay={n_lindsay}, Rogini={n_rogini}")

    # Select top genes (skip if n_top is None — caller already filtered)
    if n_top is not None:
        print(f"  Selecting top {n_top} genes by {filter_method}...")
        rna_data = _select_top_genes(rna_data, n_top=n_top, method=filter_method)
    n_genes = rna_data.shape[1]
    n_samples = rna_data.shape[0]
    print(f"  Data: {n_samples} samples x {n_genes} genes")

    # ---- log-CPM normalization ----
    print("  Applying log-CPM normalization...")
    lib_sizes = np.asarray(rna_data.sum(axis=1), dtype=np.float64).ravel()
    zero_lib = lib_sizes == 0
    if zero_lib.any():
        print(f"  {zero_lib.sum()} fish have zero total counts — "
              f"setting library size to 1 to avoid division by zero")
        lib_sizes[zero_lib] = 1.0
    cpm = rna_data / (lib_sizes[:, None] / 1e6)
    expr_vst = np.log2(cpm + 1.0)
    # Fish with zero counts produce 0/1=0 → log2(0+1)=0, which is correct
    expr_vst = np.asarray(expr_vst.T, dtype=np.float64)  # genes × samples
    print(f"  log-CPM complete: {expr_vst.shape[0]} genes x "
          f"{expr_vst.shape[1]} samples")

    # ---- Check batch effect BEFORE ComBat ----
    batch_numeric = (design_data['Batch'].values == 'Rogini').astype(float)
    if np.std(batch_numeric) > 0:
        pca_before = _pca_correlation(expr_vst.T, batch_numeric)
        print(f"  PC1-Batch correlation before: {pca_before[0]:.4f}")
        print(f"  PC2-Batch correlation before: {pca_before[1]:.4f}")
    else:
        pca_before = (0.0, 0.0)
        print("  Single batch — skipping PCA correlation")

    # ---- Determine ComBat design matrix ----
    if combat_formula is not None:
        print(f"  ComBat design: {combat_formula}")
    else:
        # Check if any lakes exist in both batches
        lake_batch_table = design_data.groupby(
            ['Lake', 'Batch'], observed=True).size().unstack(fill_value=0)
        n_both = sum((lake_batch_table > 0).sum(axis=1) == 2)
        if n_both > 0:
            print(f"  {n_both} lakes in both batches — protecting lake effects (~ Lake)")
            combat_formula = "~ Lake"
        else:
            print("  No lake overlap across batches — using null design")

    # ---- Build design matrix for biological effects to preserve ----
    # pycombat.Combat.fit_transform(Y, b, X, C):
    #   Y: (n_samples, n_features) — expression data
    #   b: (n_samples,) — batch labels
    #   X: (n_samples, mx) — variables to PRESERVE (e.g. Lake)
    #   C: (n_samples, mc) — variables to REMOVE (not used here)
    batch_labels = design_data['Batch'].values

    X_preserve = None
    if combat_formula is not None:
        import re
        terms = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', combat_formula)
        terms = [t for t in terms if t != '~']
        if terms:
            preserve_cols = []
            for term in terms:
                if term in design_data.columns:
                    encoded = pd.get_dummies(
                        design_data[term].astype(str), drop_first=True,
                        dtype=float,
                    )
                    for col in encoded.columns:
                        preserve_cols.append(encoded[col].values)
            if preserve_cols:
                X_preserve = np.column_stack(preserve_cols)

    # ---- Apply ComBat (skip if only one batch) ----
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        print(f"  Skipping ComBat — only {len(unique_batches)} batch present "
              f"({unique_batches[0]})")
        expr_corrected = np.asarray(expr_vst, dtype=np.float64)
        pca_after = pca_before
        pc1_reduction = 0.0
    else:
        # pycombat expects Y as (n_samples, n_features), so transpose from
        # genes×samples to samples×genes.
        print("  Applying ComBat...")
        Y = expr_vst.T  # samples × genes

        combat = Combat()
        expr_corrected = combat.fit_transform(
            Y=Y,
            b=batch_labels,
            X=X_preserve,
        )
        # Transpose back to genes × samples
        expr_corrected = np.asarray(expr_corrected.T, dtype=np.float64)
        print("  ComBat complete")

        # ---- Check batch effect AFTER ComBat ----
        pca_after = _pca_correlation(expr_corrected.T, batch_numeric)

        if abs(pca_before[0]) > 0.001:
            pc1_reduction = 100 * (1 - abs(pca_after[0]) / abs(pca_before[0]))
        else:
            pc1_reduction = float('nan')

        print(f"  PC1-Batch correlation after:  {pca_after[0]:.4f}")
        print(f"  PC2-Batch correlation after:  {pca_after[1]:.4f}")
        if not np.isnan(pc1_reduction):
            print(f"  PC1 reduction: {pc1_reduction:.1f}%")

        if not np.isnan(pc1_reduction) and abs(pca_after[0]) < 0.1 and pc1_reduction > 80:
            print("  [OK] EXCELLENT: Batch effects successfully removed")
        elif not np.isnan(pc1_reduction) and abs(pca_after[0]) < 0.2 and pc1_reduction > 60:
            print("  [OK] GOOD: Substantial batch effect reduction")
        else:
            print("  [WARNING] Some batch effects may remain — inspect PCA plots")

    # ---- Save outputs ----
    expr_path = os.path.join(output_dir, 'expression_batch_corrected.csv')
    pd.DataFrame(expr_corrected).to_csv(expr_path)
    print(f"  Saved: {expr_path}")

    design_path = os.path.join(output_dir, 'design_with_batch.csv')
    design_data.to_csv(design_path, index=False)
    print(f"  Saved: {design_path}")

    validation = pd.DataFrame({
        'Metric': [
            'PC1_Batch_Before', 'PC1_Batch_After',
            'PC2_Batch_Before', 'PC2_Batch_After',
            'PC1_Reduction_Percent',
            'Total_Samples', 'Lindsay_Samples', 'Rogini_Samples',
            'Genes_Analyzed',
            'Transformation',
            'Normalization_Applied',
        ],
        'Value': [
            round(pca_before[0], 4), round(pca_after[0], 4),
            round(pca_before[1], 4), round(pca_after[1], 4),
            round(pc1_reduction, 1) if not np.isnan(pc1_reduction) else 'NA',
            n_samples, n_lindsay, n_rogini,
            n_genes,
            'log-CPM',
            'None — apply your own normalization before downstream analysis',
        ],
    })
    validation_path = os.path.join(output_dir, 'batch_correction_validation.csv')
    validation.to_csv(validation_path, index=False)
    print(f"  Saved: {validation_path}")

    return {
        'expr_corrected': expr_corrected,
        'design': design_data,
        'validation': validation,
        'pc1_before': pca_before[0],
        'pc1_after': pca_after[0],
        'pc1_reduction': pc1_reduction,
        'gene_names': list(rna_data.columns),
    }


def _pca_correlation(expr_samples_x_features, batch_numeric):
    """Compute Pearson correlation of PC1, PC2 with batch assignment.

    Returns (pc1_corr, pc2_corr).  Returns (nan, nan) if PCA cannot be
    computed (e.g. all-NaN rows from fish with zero library size).
    """
    X = np.asarray(expr_samples_x_features, dtype=np.float64)

    # Drop genes (columns) with NaN/inf FIRST — a single bad gene in every
    # sample would otherwise cause all samples to be dropped.
    finite_cols = np.isfinite(X).all(axis=0)
    n_bad_genes = int((~finite_cols).sum())
    if n_bad_genes > 0:
        print(f"  _pca_correlation: dropping {n_bad_genes} genes with "
              f"non-finite values (likely zero-variance after ComBat)")
    X = X[:, finite_cols]

    # Drop constant genes (zero std)
    gene_std = np.std(X, axis=0)
    variable = gene_std > 0
    n_constant = int((~variable).sum())
    if n_constant > 0:
        print(f"  _pca_correlation: dropping {n_constant} constant genes")
    X = X[:, variable]

    if X.shape[1] < 2:
        print(f"  _pca_correlation: only {X.shape[1]} variable genes "
              f"(need ≥2) — returning nan")
        return float('nan'), float('nan')

    # Drop samples (rows) that still have NaN/inf after gene filtering.
    # A fish with zero total counts produces inf in CPM → inf in log-CPM.
    finite_rows = np.isfinite(X).all(axis=1)
    n_bad_samples = int((~finite_rows).sum())
    if n_bad_samples > 0:
        print(f"  _pca_correlation: dropping {n_bad_samples} samples with "
              f"non-finite expression (likely zero library size)")

    X = X[finite_rows, :]
    batch_numeric = np.asarray(batch_numeric, dtype=np.float64)[finite_rows]

    if X.shape[0] < 3:
        print(f"  _pca_correlation: only {X.shape[0]} valid samples "
              f"(need ≥3) — returning nan")
        return float('nan'), float('nan')

    if X.shape[1] < 2:
        print(f"  _pca_correlation: only {X.shape[1]} variable genes "
              f"(need ≥2) — returning nan")
        return float('nan'), float('nan')

    # Center
    X = X - X.mean(axis=0)

    try:
        if X.shape[1] > X.shape[0]:
            gram = X @ X.T
            eigvals, eigvecs = np.linalg.eigh(gram)
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
        else:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            eigvecs = U
    except np.linalg.LinAlgError:
        print(f"  _pca_correlation: SVD/eigh failed — returning nan")
        return float('nan'), float('nan')

    if eigvecs.shape[1] < 2 or np.std(batch_numeric) == 0:
        return float('nan'), float('nan')

    pc1_corr = np.corrcoef(eigvecs[:, 0], batch_numeric)[0, 1]
    pc2_corr = np.corrcoef(eigvecs[:, 1], batch_numeric)[0, 1]
    return float(pc1_corr), float(pc2_corr)


def batch_correct_hk(metadata_path, transcriptome_path,
                     morphology_path=None, output_dir='results_batch_corrected',
                     n_top=10000, filter_method='expression'):
    """Run full batch correction pipeline for HK transcriptome.

    Parameters
    ----------
    metadata_path : str
        Path to 1.Metadata.csv.
    transcriptome_path : str
        Path to 7.HKTranscriptome.csv.
    morphology_path : str or None
        Path to 2.Morphology.csv for sex information (optional).
        The original R script uses 3.sex.csv, but sex is only carried
        in the output design file — it is NOT used in the ComBat model.
        Safe to skip if unavailable.
    output_dir : str
        Root output directory. Subdirectories hk/source_lakes/ and
        hk/recipient_lakes/ are created automatically.
    n_top : int
        Number of top genes to retain for batch correction (default 10000).
    filter_method : str
        'expression' or 'variance' for selecting top genes.

    Returns
    -------
    dict with keys 'source_lakes', 'recipient_lakes', each mapping to
    the dict returned by run_combat.
    """
    print("=" * 70)
    print("COMBAT BATCH CORRECTION — HK Transcriptome")
    print("=" * 70)

    # ---- Load metadata ----
    print("\nLoading metadata...")
    meta = pd.read_csv(metadata_path)
    # Drop leading empty column if present
    if meta.columns[0] == '' or meta.columns[0].startswith('Unnamed'):
        meta = meta.drop(columns=[meta.columns[0]])
    print(f"  {len(meta)} samples")

    # ---- Optionally merge sex ----
    if morphology_path is not None and os.path.exists(morphology_path):
        print(f"Loading morphology (sex) from {morphology_path}...")
        morph = pd.read_csv(morphology_path)
        if morph.columns[0] == '' or morph.columns[0].startswith('Unnamed'):
            morph = morph.drop(columns=[morph.columns[0]])
        if 'Sex_f_m_NA' in morph.columns:
            meta = meta.merge(
                morph[['Fish_ID', 'Sex_f_m_NA']],
                on='Fish_ID', how='left',
            )
            meta['sex'] = meta['Sex_f_m_NA'].fillna('imm')
            print(f"  Merged sex info for {meta['sex'].notna().sum()} samples")
    else:
        meta['sex'] = 'NA'
        print("  No morphology file — sex set to 'NA'")

    # ---- Load transcriptome ----
    print(f"\nLoading transcriptome: {transcriptome_path}")
    rna_raw = pd.read_csv(transcriptome_path)
    # Drop leading empty column if present
    if rna_raw.columns[0] == '' or str(rna_raw.columns[0]).startswith('Unnamed'):
        rna_raw = rna_raw.drop(columns=[rna_raw.columns[0]])
    # Ensure Fish_ID column exists
    if rna_raw.columns[0] != 'Fish_ID':
        raise ValueError(f"Expected Fish_ID as first column, got: {rna_raw.columns[0]}")
    print(f"  HK raw: {rna_raw.shape[0]} samples x {rna_raw.shape[1]} columns (incl. Fish_ID)")

    # ---- Align metadata and transcriptome by Fish_ID ----
    merged = meta.merge(rna_raw, on='Fish_ID', how='inner')
    print(f"  Aligned: {merged.shape[0]} samples matched")
    if merged.shape[0] == 0:
        raise RuntimeError("No samples matched between metadata and transcriptome — check Fish_ID values")

    # Split expression columns from metadata columns
    meta_cols = list(meta.columns)
    expr_cols = [c for c in merged.columns if c not in meta_cols]
    meta_aligned = merged[meta_cols].reset_index(drop=True)
    rna_hk = merged[expr_cols].reset_index(drop=True)
    # Fill NaN with 0 — missing counts are treated as zero expression
    if rna_hk.isna().any().any():
        nan_count = rna_hk.isna().sum().sum()
        print(f"  Filling {nan_count} NaN values with 0")
        rna_hk = rna_hk.fillna(0)
    print(f"  HK aligned: {rna_hk.shape[0]} samples x {rna_hk.shape[1]} genes")

    # ---- Select top genes from FULL dataset BEFORE splitting ----
    # The original R script selects top 10k independently per group
    # (source vs recipient), which produces different gene sets.  We
    # select once on the pooled data so both groups share identical
    # gene columns — required for combining into a single pipeline CSV.
    print(f"\nSelecting top {n_top} genes by {filter_method} "
          f"from pooled data ({rna_hk.shape[0]} samples)...")
    rna_hk, kept_genes = _select_top_genes_and_track(
        rna_hk, n_top=n_top, method=filter_method)
    print(f"  Retained {len(kept_genes)}/{rna_hk.shape[1]} genes "
          f"(actually {len(kept_genes)} — some may have been dropped)")

    # ---- Drop near-zero-expression genes (prevent ComBat divide-by-zero) ----
    # When keeping all genes, some have extremely low expression (1-2 counts
    # in a handful of samples).  After log-CPM, their per-batch std ≈ 0,
    # and ComBat does Z/sigma → inf for every sample.
    # Filter on pooled data so source/recipient have identical gene sets.
    # Drop genes with <100 total counts (≈0.05 counts/sample) AND genes
    # with zero variance in either batch.
    total_expr = np.asarray(rna_hk.sum(axis=0)).ravel()
    batch_all = _assign_batch(meta_aligned['Year'])
    is_lindsay = batch_all == 'Lindsay'
    is_rogini = batch_all == 'Rogini'
    var_l = np.asarray(rna_hk.loc[is_lindsay, :].var(axis=0)).ravel()
    var_r = np.asarray(rna_hk.loc[is_rogini, :].var(axis=0)).ravel()
    keep = (total_expr >= 100) & (var_l > 0) & (var_r > 0)
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} near-zero-expression genes "
              f"(<100 total counts or zero within-batch variance)")
        rna_hk = rna_hk.loc[:, keep]

    # ---- Split source vs recipient ----
    is_destination = meta_aligned['Destination'].astype(bool).values
    meta_src = meta_aligned[~is_destination].copy()
    meta_rec = meta_aligned[is_destination].copy()
    rna_hk_src = rna_hk.loc[~is_destination].copy()
    rna_hk_rec = rna_hk.loc[is_destination].copy()

    # Drop samples with missing lake
    keep_src = ~meta_src['Lake'].isna()
    meta_src = meta_src[keep_src].reset_index(drop=True)
    rna_hk_src = rna_hk_src[keep_src.values].reset_index(drop=True)

    # For recipient: remove G Lake (incomplete years)
    keep_rec = (meta_rec['Lake'] != 'G Lake') & (~meta_rec['Lake'].isna())
    meta_rec = meta_rec[keep_rec].reset_index(drop=True)
    rna_hk_rec = rna_hk_rec[keep_rec.values].reset_index(drop=True)

    # ---- Build design matrices ----
    design_src = meta_src[['Lake', 'Year', 'sex']].copy()
    design_src['Year'] = design_src['Year'].astype(str).astype('category')
    design_src['Lake'] = design_src['Lake'].astype(str).astype('category')
    design_src['sex'] = design_src['sex'].astype(str).astype('category')

    design_rec = meta_rec[['Lake', 'Year', 'sex']].copy()
    design_rec['Year'] = design_rec['Year'].astype(str).astype('category')
    design_rec['Lake'] = design_rec['Lake'].astype(str).astype('category')
    design_rec['sex'] = design_rec['sex'].astype(str).astype('category')

    print(f"\nSource lakes:     {rna_hk_src.shape[0]} samples, "
          f"{len(design_src['Lake'].unique())} lakes")
    print(f"Recipient lakes:  {rna_hk_rec.shape[0]} samples, "
          f"{len(design_rec['Lake'].unique())} lakes")

    # ---- Process source lakes ----
    print("\n" + "-" * 70)
    print("SOURCE LAKES")
    print("-" * 70)
    res_src = run_combat(
        rna_data=rna_hk_src,
        design_data=design_src,
        output_dir=os.path.join(output_dir, 'hk', 'source_lakes'),
        combat_formula="~ Lake",
        n_top=None,  # genes already selected from pooled data
    )

    # ---- Process recipient lakes ----
    print("\n" + "-" * 70)
    print("RECIPIENT LAKES")
    print("-" * 70)
    res_rec = run_combat(
        rna_data=rna_hk_rec,
        design_data=design_rec,
        output_dir=os.path.join(output_dir, 'hk', 'recipient_lakes'),
        combat_formula=None,  # auto-detect
        n_top=None,  # genes already selected from pooled data
    )

    # ---- Build combined pipeline-ready CSV (Fish_ID × genes) ----
    # The per-group CSVs saved by run_combat are genes×samples without
    # sample or gene labels.  Reconstruct a single CSV that matches the
    # format expected by SticklebackData (Fish_ID col + gene cols).
    print("\n" + "-" * 70)
    print("BUILDING PIPELINE-READY COMBINED CSV")
    print("-" * 70)

    # Gene sets are identical by construction — selected once from the
    # pooled dataset before splitting into source/recipient.
    gene_names = res_src['gene_names']

    # Reconstruct samples×genes DataFrames with Fish_ID index
    fish_ids_src = meta_src['Fish_ID'].values
    fish_ids_rec = meta_rec['Fish_ID'].values

    # run_combat returns genes×samples; transpose to samples×genes
    expr_src_df = pd.DataFrame(
        res_src['expr_corrected'].T,
        index=fish_ids_src,
        columns=gene_names,
    )
    expr_rec_df = pd.DataFrame(
        res_rec['expr_corrected'].T,
        index=fish_ids_rec,
        columns=gene_names,
    )

    # Combine source + recipient
    combined = pd.concat([expr_src_df, expr_rec_df], axis=0)
    combined.index.name = 'Fish_ID'
    combined = combined.reset_index()  # Fish_ID becomes first column

    # Save
    pipeline_csv = os.path.join(output_dir, 'hk',
                                'expression_batch_corrected_pipeline.csv')
    combined.to_csv(pipeline_csv, index=False)
    print(f"  Combined: {combined.shape[0]} samples × "
          f"{combined.shape[1] - 1} genes")
    print(f"  Saved: {pipeline_csv}")

    print("\n" + "=" * 70)
    print("BATCH CORRECTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_dir}/")
    print(f"  hk/source_lakes/expression_batch_corrected.csv")
    print(f"  hk/recipient_lakes/expression_batch_corrected.csv")
    print(f"  hk/expression_batch_corrected_pipeline.csv  ← pipeline input")

    return {
        'source_lakes': res_src,
        'recipient_lakes': res_rec,
        'pipeline_csv': pipeline_csv,
    }
