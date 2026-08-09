#!/usr/bin/env python3
"""
Full RoL pipeline with file-based checkpointing.

Each step saves intermediate results to the output directory.
If a checkpoint exists, that step is skipped unless --force is passed.

Graphs can be many GBs — intermediate files are written to disk so memory
can be freed between steps and the pipeline can be resumed after interruption.

Steps:
  0. Batch correction (optional)            → results_batch_corrected/
  1. Load data & build expression matrices  → matrices.pkl
  2. Build co-expression graphs             → graphs/ + graphs_manifest.pkl
  3. Train VQGNN model                      → model.pt
  4. Generate lake embeddings               → embeddings.pkl
  5. Build gene-VQ mappings                 → gene_mappings.pkl
  6. Temporal slope analysis                → slopes.pkl
  7. Hierarchical clustering                → clustering.pkl
  8. Label-permutation test                 → label_perm.pkl
  9. Generate figures                       → figures/ directory
 10. Post-processing (stats + tables)       → stats.json, enrichment CSVs

Usage:
  python pipeline.py --data-dir ./data --output-dir ./output
  python pipeline.py --data-dir ./data --output-dir ./output --force
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
from tqdm import tqdm

# Add project root to path so `rol` is importable from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def checkpoint_path(output_dir, filename):
    return os.path.join(output_dir, filename)


def exists(path):
    return os.path.exists(path)


def save_pickle(obj, path):
    """Write pickle atomically: dump to a pid-suffixed temp file, then rename.

    A concurrent reader can never observe a half-written checkpoint: the file
    appears at ``path`` only once fully written (``os.replace`` is atomic on
    the same filesystem). The pid suffix avoids two writers colliding on the
    same temp name when multiple Modal entrypoints share an output volume.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _joint_path(args, filename):
    """Insert ``_joint`` before the extension if training in joint mode."""
    if args.train_joint:
        base, ext = os.path.splitext(filename)
        return f"{base}_joint{ext}"
    return filename


# DEAD CODE (unused): _validate_graphs — all call sites are commented out
# with "too slow" notes (loads every .pt file).  Degenerate detection now
# happens inline during graph loading in _preload_all_graphs / _load_batch.
# Kept as reference for manual ad-hoc validation.
#
# def _validate_graphs(graphs, label="graphs", graphs_dir=None):
#     """Check for degenerate/empty graphs and warn if any are found.
#
#     If ``graphs_dir`` is provided and contains ``degenerate_manifest.json``
#     (saved by ``build_all``), validation is instant — no graph files are
#     loaded.  Otherwise falls back to checking each entry individually.
#     """
#     from surge.graphs import CoexpressionGraphBuilder
#     import json as _json
#
#     # Fast path: pre-computed degenerate manifest from build_all
#     if graphs_dir is not None:
#         manifest_path = os.path.join(graphs_dir, 'degenerate_manifest.json')
#         if os.path.exists(manifest_path):
#             with open(manifest_path) as fh:
#                 degenerate_keys = set(_json.load(fh))
#             n_total = len(graphs)
#             n_degen = len(degenerate_keys)
#             pct = 100 * n_degen / n_total if n_total > 0 else 0
#             if n_degen > 0:
#                 print(f"\n  Graph validation ({label}): {n_degen}/{n_total} "
#                       f"({pct:.1f}%) degenerate/empty (from manifest)")
#                 for i, key in enumerate(sorted(degenerate_keys)[:10]):
#                     print(f"    [{i+1}] {key}")
#                 if n_degen > 10:
#                     print(f"    ... and {n_degen - 10} more")
#             else:
#                 print(f"  Graph validation ({label}): all {n_total} ok — no "
#                       f"degenerate/empty graphs")
#             return
#
#     # Slow path: check each entry individually
#     degenerate = {}
#     for key, val in list(graphs.items()):
#         if CoexpressionGraphBuilder.is_degenerate(val):
#             degenerate[key] = val
#
#     n_total = len(graphs)
#     n_degen = len(degenerate)
#     pct = 100 * n_degen / n_total if n_total > 0 else 0
#
#     if n_degen > 0:
#         print(f"\n  Graph validation ({label}): {n_degen}/{n_total} "
#               f"({pct:.1f}%) degenerate/empty")
#         # List first 10 degenerate keys
#         for i, key in enumerate(sorted(degenerate.keys(), key=str)[:10]):
#             print(f"    [{i+1}] {key}")
#         if n_degen > 10:
#             print(f"    ... and {n_degen - 10} more")
#         if pct > 10:
#             print(f"  WARNING: >10% of {label} are degenerate. "
#                   f"Check expression data quality.")
#     else:
#         print(f"  Graph validation ({label}): all {n_total} ok — no "
#               f"degenerate/empty graphs")
#
#     return degenerate, n_degen


def _save_model_kwargs(output_dir, model_filename, n_genes, cfg, n_lakes):
    """Save model constructor kwargs so Modal workers can reconstruct the model."""
    kwargs_path = os.path.join(
        output_dir,
        model_filename.replace('.pt', '_kwargs.pkl')
    )
    if os.path.exists(kwargs_path):
        return  # already saved
    kwargs = {
        'n_nodes': n_genes,
        'in_channels': cfg.get('in_channels', 64),
        'hidden_channels': cfg.get('hidden_channels', 64),
        'out_channels': cfg.get('out_channels', 16),
        'num_layers': cfg.get('num_layers', 3),
        'dropout': cfg.get('dropout', 0.3),
        'codebook_channels': cfg.get('codebook_channels', 16),
        'codebook_size': cfg.get('codebook_size', 100),
        'decoder_channels': cfg.get('decoder_channels', 256),
        'n_lakes': n_lakes,
    }
    import pickle
    with open(kwargs_path, 'wb') as f:
        pickle.dump(kwargs, f)
    print(f"  Saved model kwargs to {kwargs_path}")


def _build_vqgnn(cfg, n_genes, n_lakes=None):
    """Build a VQGNN model from config dict and gene/lake counts.

    Extracted to avoid duplicating the 12-parameter constructor call across
    ``_train_models_batched`` and ``step_train_model``.
    """
    from surge.vqgnn import VQGNN

    return VQGNN(
        n_nodes=n_genes,
        in_channels=cfg.get('in_channels', 64),
        hidden_channels=cfg.get('hidden_channels', 64),
        out_channels=cfg.get('out_channels', 16),
        num_layers=cfg.get('num_layers', 3),
        dropout=cfg.get('dropout', 0.3),
        codebook_channels=cfg.get('codebook_channels', 16),
        codebook_size=cfg.get('codebook_size', 100),
        decoder_channels=cfg.get('decoder_channels', 256),
        n_lakes=n_lakes,
    )


def _resolve_graph_radii(key, radii_dict, bundled_r, n_nodes):
    """Resolve noise radii for a single graph, with fallback chain.

    Priority: radii_dict entry > bundled radii from .pt file > ones(n_nodes).
    """
    import torch

    r = radii_dict.get(str(key))
    if r is None and bundled_r is not None:
        if isinstance(bundled_r, torch.Tensor):
            r = bundled_r.detach().cpu().numpy()
        else:
            r = bundled_r
    if r is None:
        r = np.ones(n_nodes, dtype=np.float32)
    elif isinstance(r, (str, os.PathLike)):
        r = np.load(os.fspath(r))
    elif isinstance(r, torch.Tensor):
        r = r.detach().cpu().numpy()
    return np.asarray(r, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 0: Batch correction (optional)
# ---------------------------------------------------------------------------

def step_batch_correct(args):
    """Run batch correction and return path to pipeline-ready CSV.

    The returned CSV is used as the transcriptome input for subsequent
    pipeline steps, replacing the raw file.
    """
    from surge.batch_correction import batch_correct_hk

    output_dir = args.batch_correct_output or os.path.join(
        args.output_dir, 'results_batch_corrected'
    )

    pipeline_csv = os.path.join(
        output_dir, 'hk', 'expression_batch_corrected_pipeline.csv'
    )

    # Check if already done — look for the pipeline-ready combined CSV
    if not args.force and os.path.exists(pipeline_csv):
        print("[Step 0] Batch correction already done, skipping.")
        print(f"  Using: {pipeline_csv}")
        return pipeline_csv

    print("[Step 0] Running log-CPM + ComBat batch correction on HK "
          "transcriptome...")
    n_top = args.batch_n_genes if args.batch_n_genes > 0 else None
    result = batch_correct_hk(
        metadata_path=args.batch_metadata or args.metadata,
        transcriptome_path=args.batch_transcriptome,
        morphology_path=args.batch_morphology or args.morphology,
        output_dir=output_dir,
        n_top=n_top,
    )
    bc_csv = result['pipeline_csv']
    print(f"[Step 0] Batch correction complete.")
    print(f"  Pipeline input: {bc_csv}")
    return bc_csv


# ---------------------------------------------------------------------------
# Step 1: Load data & build expression matrices
# ---------------------------------------------------------------------------

def step_load_data(args, cfg):
    from surge.data import SticklebackData

    matrices_path = checkpoint_path(args.output_dir, 'matrices.pkl')
    data_path = checkpoint_path(args.output_dir, 'data.pkl')

    if not args.force and exists(matrices_path) and exists(data_path):
        cached = load_pickle(matrices_path)
        if isinstance(cached, dict) and '__meta__' in cached:
            cached_strats = cached['__meta__'].get('stratifications', [])
            if set(cached_strats) == set(cfg['stratifications']):
                print("[Step 1] Loading cached matrices and data...")
                del cached['__meta__']
                return cached, load_pickle(data_path)
            print(f"[Step 1] Cached stratifications {cached_strats} "
                  f"differ from current {cfg['stratifications']} — rebuilding")
        elif isinstance(cached, dict):
            # Backward-compatible: old cache without metadata, assume valid
            print("[Step 1] Loading cached matrices and data (legacy)...")
            return cached, load_pickle(data_path)

    print("[Step 1] Loading CSVs and building expression matrices...")
    # When --batch-correct ran, args.transcriptome points at the log2(CPM+1)
    # ComBat CSV — tell the loader so it inverts to linear before row-normalizing.
    input_scale = 'log2cpm' if getattr(args, 'batch_correct', False) else 'linear'
    sd = SticklebackData(
        transcriptome_path=args.transcriptome,
        metadata_path=args.metadata,
        morphology_path=args.morphology or None,
        infection_path=args.infection or None,
        input_scale=input_scale,
    )

    matrices = {}
    for by in cfg['stratifications']:
        print(f"  Stratifying by: {by}")
        by_matrices = sd.build_all_matrices(by=by)
        overlap = set(matrices).intersection(by_matrices)
        if overlap:
            sample = sorted(map(str, overlap))[:5]
            raise ValueError(
                f"Duplicate matrix keys detected while merging '{by}'. "
                f"Overlapping keys (sample): {sample}. "
                "Use stratification-specific key prefixes to avoid overwrite."
            )
        matrices.update(by_matrices)

    save_pickle({**matrices, '__meta__': {'stratifications': cfg['stratifications']}}, matrices_path)
    save_pickle(sd, data_path)
    print(f"  Saved {len(matrices)} matrices to {matrices_path}")
    return matrices, sd


# ---------------------------------------------------------------------------
# Step 2: Build co-expression graphs
# ---------------------------------------------------------------------------

def step_build_graphs(args, cfg, matrices):
    from surge.graphs import CoexpressionGraphBuilder

    graphs_manifest_path = checkpoint_path(args.output_dir, 'graphs_manifest.pkl')
    graphs_dir = checkpoint_path(args.output_dir, 'graphs')
    radii_path = checkpoint_path(args.output_dir, 'radii.pkl')
    legacy_graphs_path = checkpoint_path(args.output_dir, 'graphs.pkl')

    if not args.force and exists(graphs_manifest_path) and exists(radii_path):
        print ("[Step 2] Checking cached graph manifest...")
        cached_graphs = load_pickle(graphs_manifest_path)
        missing = set(matrices) - set(cached_graphs)
        if not missing:
            # Verify cached graphs have the expected number of nodes
            # (important when --top-n-genes filtered the gene set)
            expected_n = cfg.get('n_genes')
            if expected_n:
                # Pick a NON-degenerate sample key: the first manifest key may
                # be a degenerate bundle (a dict, no num_nodes), which would
                # crash the node-count validation below.
                import torch as _torch
                from surge.graphs import CoexpressionGraphBuilder as _CGB
                sample_key = None
                sample_val = None
                for _k, _v in cached_graphs.items():
                    if not _CGB.is_degenerate(_v):
                        sample_key, sample_val = _k, _v
                        break
                if sample_val is not None:
                    # Resolve actual graph to get num_nodes
                    if isinstance(sample_val, (str, os.PathLike)):
                        g = _torch.load(os.fspath(sample_val),
                                        map_location='cpu',
                                        weights_only=False)
                        if isinstance(g, dict) and 'graphs' in g:
                            g = g['graphs'][0]
                        elif isinstance(g, (list, tuple)):
                            g = g[0]
                    elif isinstance(sample_val, (list, tuple)):
                        g = sample_val[0]
                    else:
                        g = sample_val
                    cached_n = (g.num_nodes if hasattr(g, 'num_nodes')
                                else g.x.shape[0] if hasattr(g, 'x')
                                else g.edge_index.max().item() + 1)
                    if cached_n != expected_n:
                        print(f"[Step 2] Cached graphs have {cached_n} nodes, "
                              f"expected {expected_n} — rebuilding")
                        args.force = True  # force rebuild for this step
                        # fall through to rebuild below
                    else:
                        print("[Step 2] Loading cached graph manifest...")
                        # _validate_graphs skipped — too slow (loads every .pt)
                        return cached_graphs, load_pickle(radii_path)
                else:
                    print("[Step 2] Loading cached graph manifest "
                          "(all keys degenerate — no node-count to validate)...")
                    return cached_graphs, load_pickle(radii_path)
            else:
                print("[Step 2] Loading cached graph manifest...")
                # _validate_graphs skipped — too slow (loads every .pt)
                return cached_graphs, load_pickle(radii_path)
        else:
            # Partial manifest: for each missing key, check whether the .pt
            # bundle already exists on disk (e.g. from a killed prior run).
            # If it does, adopt it by path without torching-loading anything;
            # load the radii sidecar which is tiny.  Only build from scratch
            # the keys whose bundles are genuinely absent.
            cached_radii = load_pickle(radii_path)
            from surge.graphs import CoexpressionGraphBuilder
            found, not_found = 0, 0
            for key in missing:
                stratification = CoexpressionGraphBuilder._infer_stratification(key)
                safe = CoexpressionGraphBuilder._safe_key(key)
                bundle_path = CoexpressionGraphBuilder._find_bundle(
                    graphs_dir, stratification, safe)
                if os.path.exists(bundle_path):
                    cached_graphs[key] = bundle_path
                    sidecar = CoexpressionGraphBuilder._bundle_radii_path(
                        graphs_dir, stratification, safe)
                    if os.path.exists(sidecar):
                        cached_radii[key] = np.load(sidecar).astype(np.float32)
                    found += 1
                else:
                    not_found += 1
            if not_found > 0:
                print(f"[Step 2] Graph manifest: {found} keys recovered from "
                      f"disk, {not_found} genuinely missing — building")
                missing_matrices = {k: matrices[k] for k in missing
                                    if not os.path.exists(
                        CoexpressionGraphBuilder._find_bundle(
                            graphs_dir,
                            CoexpressionGraphBuilder._infer_stratification(k),
                            CoexpressionGraphBuilder._safe_key(k)))}
                builder = CoexpressionGraphBuilder(
                    n_eigencomponents=cfg.get('n_eigencomponents', 128),
                    reconstruction_levels=cfg.get('reconstruction_levels',
                                                   [4, 8, 32, 64, 127]),
                    device='cpu',
                )
                new_graphs, new_radii = builder.build_all(
                    missing_matrices,
                    output_dir=graphs_dir,
                    keep_in_memory=False,
                    reuse_existing=False,
                )
                cached_graphs.update(new_graphs)
                cached_radii.update(new_radii)
            else:
                print(f"[Step 2] All {found} missing keys recovered from "
                      f"disk — no rebuild needed")
            save_pickle(cached_graphs, graphs_manifest_path)
            save_pickle(cached_radii, radii_path)
            return cached_graphs, cached_radii

    # Backward-compatible load for old checkpoint layout.
    if not args.force and exists(legacy_graphs_path) and exists(radii_path):
        legacy_graphs = load_pickle(legacy_graphs_path)
        print("[Step 2] Loading legacy cached graphs.pkl...")
        # _validate_graphs skipped — too slow (loads every .pt)
        return legacy_graphs, load_pickle(radii_path)

    # If no manifest exists, scan disk for partial graph builds so we can
    # resume from where a previous (killed) run left off.
    if not args.force and exists(graphs_dir) and not exists(graphs_manifest_path):
        print ("[Step 2] No graph manifest found, scanning disk for existing graph bundles...")
        import glob as _glob
        from surge.graphs import CoexpressionGraphBuilder
        existing_graphs = {}
        existing_radii = {}
        # Build a reverse-map: expected filename → key
        name_to_key = {}
        for key in matrices:
            strat = CoexpressionGraphBuilder._infer_stratification(key)
            safe = CoexpressionGraphBuilder._safe_key(key)
            name_to_key[f"{strat}_{safe}"] = key
        # Scan disk for .pt bundles (filename match only — no loading)
        pattern = os.path.join(graphs_dir, "*.pt")
        found = 0
        for fpath in tqdm(_glob.glob(pattern), desc="  Scanning graph bundles", unit="file"):
            basename = os.path.splitext(os.path.basename(fpath))[0]
            key = name_to_key.get(basename)
            if key is None:
                continue  # stale or unmatched file
            existing_graphs[key] = fpath
            # Try to load matching radii
            radii_sidecar = os.path.join(
                graphs_dir,
                f"{basename}.radii.npy"
            )
            if os.path.exists(radii_sidecar):
                try:
                    existing_radii[key] = np.load(radii_sidecar)
                except Exception:
                    pass
            found += 1
        if found > 0:
            print(f"[Step 2] Found {found} existing graph bundles on disk "
                  f"— saving partial manifest for resumption")
            save_pickle(existing_graphs, graphs_manifest_path)
            save_pickle(existing_radii if existing_radii else {},
                        radii_path)
            # Reload through normal path → will hit partial-manifest path above
            return step_build_graphs(args, cfg, matrices)

    # No cached graphs at all — build everything from scratch.
    print("[Step 2] Building co-expression graphs (resumable, disk-backed)...")
    builder = CoexpressionGraphBuilder(
        n_eigencomponents=cfg.get('n_eigencomponents', 128),
        reconstruction_levels=cfg.get('reconstruction_levels', [4, 8, 32, 64, 127]),
        device='cpu',
    )

    graphs, radii = builder.build_all(
        matrices,
        output_dir=graphs_dir,
        keep_in_memory=False,
        reuse_existing=not args.force,
    )

    save_pickle(graphs, graphs_manifest_path)
    save_pickle(radii, radii_path)
    print(f"  Saved graph bundles for {len(graphs)} keys in {graphs_dir}")
    print(f"  Saved graph manifest to {graphs_manifest_path}")

    # Validate: check for degenerate/empty graphs — skipped (too slow)
    # _validate_graphs(graphs, label="freshly built graphs", graphs_dir=graphs_dir)

    return graphs, radii


# ---------------------------------------------------------------------------
# Step 3: Train VQGNN model
# ---------------------------------------------------------------------------

def _preload_all_graphs(graphs, radii, noise_scale, lake_name_to_id=None):
    """Load all graph bundles from disk once, return both training formats.

    Returns a dict with keys:
      - joint: (edge_indices, target_adjs, radii_list, lake_ids)
      - sequential: list of (key, edge_index, target_adj, scaled_radii)
      - n_loaded: int
    """
    import torch

    list_of_edge_indices = []
    list_of_target_adjs = []
    radii_list = []
    lake_ids_list = []
    recon_weights_list = []
    seq_data = []  # for embedder.train()
    skipped_degenerate = []  # degenerate keys logged at end

    keys_sorted = sorted(graphs.keys())
    for key in tqdm(keys_sorted, desc="  Loading graphs", unit="graph"):
        graph_entry = graphs[key]
        bundled_r = None
        graph_list = None

        if isinstance(graph_entry, (str, os.PathLike)):
            payload = torch.load(os.fspath(graph_entry), map_location='cpu',
                                 weights_only=False)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Expected dict bundle for {key}, got {type(payload)}")
            # Degenerate sentinel: saved without a 'graphs' key when the
            # reconstruction was all-empty or all-dense (>99.9%).
            if payload.get('degenerate'):
                skipped_degenerate.append(str(key))
                continue
            if 'graphs' not in payload:
                raise KeyError(
                    f"Bundle for {key} missing 'graphs' key — "
                    f"not a degenerate sentinel (has {sorted(payload.keys())})")
            graph_list = payload['graphs']
            bundled_r = payload.get('radii')
        elif isinstance(graph_entry, (list, tuple)):
            graph_list = list(graph_entry)
        elif isinstance(graph_entry, dict) and graph_entry.get('degenerate'):
            skipped_degenerate.append(str(key))
            continue
        else:
            graph_list = [graph_entry]

        if not graph_list:
            continue

        # Resolve radii once per key — radii are per-gene and shared across
        # all reconstruction levels of the same graph.
        r = _resolve_graph_radii(key, radii, bundled_r, graph_list[0].num_nodes)
        # ORIGINAL (uncomment if _resolve_graph_radii breaks):
        # r = radii.get(str(key))
        # if r is None and bundled_r is not None:
        #     if isinstance(bundled_r, torch.Tensor):
        #         r = bundled_r.detach().cpu().numpy()
        #     else:
        #         r = bundled_r
        # if r is None:
        #     r = np.ones(graph_list[0].num_nodes, dtype=np.float32)
        # elif isinstance(r, (str, os.PathLike)):
        #     r = np.load(os.fspath(r))
        # elif isinstance(r, torch.Tensor):
        #     r = r.detach().cpu().numpy()
        # r = np.asarray(r, dtype=np.float32)
        scaled = r * float(noise_scale) if noise_scale != 0 else r.copy()

        # Train on EVERY reconstruction level (densities ramp 1% → 2% → …).
        # Level i (0-based) covers (i+1)% of edges, so its reconstruction
        # loss is divided by (i+1) to keep levels balanced.
        lake_name = str(key).split(' (')[0]
        for i, graph in enumerate(graph_list):
            # Joint format
            list_of_edge_indices.append(graph.edge_index)
            list_of_target_adjs.append(graph.target_adj)
            radii_list.append(r * float(noise_scale) if noise_scale != 0 else r.copy())
            recon_weights_list.append(1.0 / (i + 1))
            if lake_name_to_id is not None:
                lake_ids_list.append(lake_name_to_id[lake_name])

            # Sequential format
            seq_data.append((key, graph.edge_index, graph.target_adj,
                             scaled, 1.0 / (i + 1)))

    if skipped_degenerate:
        print(f"  Skipped {len(skipped_degenerate)} degenerate graph(s): "
              f"{', '.join(sorted(skipped_degenerate)[:10])}"
              f"{' ...' if len(skipped_degenerate) > 10 else ''}")
    return {
        'joint': (list_of_edge_indices, list_of_target_adjs,
                  radii_list, lake_ids_list, recon_weights_list),
        'sequential': seq_data,
        'n_loaded': len(list_of_edge_indices),
        'n_degenerate': len(skipped_degenerate),
    }


def _train_models_batched(args, cfg, graphs, radii, lake_name_to_id, n_lakes,
                          all_keys, batch_size):
    """Train sequential and joint models with batched graph loading.

    Graphs are loaded from disk in batches of ``batch_size`` to keep CPU
    memory bounded.  Both models see every graph in each epoch, but the
    training loop loads one batch, trains both models on it, then discards
    it before loading the next batch.  Optimizer state persists across
    batches.

    Returns
    -------
    seq_model, joint_model : VQGNN
    """
    import torch
    from surge.vqgnn import VQGNN

    n_genes = cfg['n_genes']
    recon_batch_size = cfg.get('recon_batch_size', 256)
    device = args.device
    n_total = len(all_keys)

    # ------------------------------------------------------------------
    # Create both models
    # ------------------------------------------------------------------
    seq_model = _build_vqgnn(cfg, n_genes, n_lakes=None)
    # ORIGINAL (uncomment if _build_vqgnn breaks):
    # seq_model = VQGNN(
    #     n_nodes=n_genes,
    #     in_channels=cfg.get('in_channels', 64),
    #     hidden_channels=cfg.get('hidden_channels', 64),
    #     out_channels=cfg.get('out_channels', 16),
    #     num_layers=cfg.get('num_layers', 3),
    #     dropout=cfg.get('dropout', 0.3),
    #     codebook_channels=cfg.get('codebook_channels', 16),
    #     codebook_size=cfg.get('codebook_size', 100),
    #     decoder_channels=cfg.get('decoder_channels', 256),
    #     n_lakes=None,
    # )
    joint_model = _build_vqgnn(cfg, n_genes, n_lakes=n_lakes)
    # ORIGINAL (uncomment if _build_vqgnn breaks):
    # joint_model = VQGNN(
    #     n_nodes=n_genes,
    #     in_channels=cfg.get('in_channels', 64),
    #     hidden_channels=cfg.get('hidden_channels', 64),
    #     out_channels=cfg.get('out_channels', 16),
    #     num_layers=cfg.get('num_layers', 3),
    #     dropout=cfg.get('dropout', 0.3),
    #     codebook_channels=cfg.get('codebook_channels', 16),
    #     codebook_size=cfg.get('codebook_size', 100),
    #     decoder_channels=cfg.get('decoder_channels', 256),
    #     n_lakes=n_lakes,
    # )
    seq_model.to(device)
    joint_model.to(device)
    print(f"[batched] Seq model device: {next(seq_model.parameters()).device}")
    print(f"[batched] Joint model device: {next(joint_model.parameters()).device}")

    # Save model kwargs for reproducibility
    seq_path = checkpoint_path(args.output_dir, 'model.pt')
    joint_path = checkpoint_path(args.output_dir, 'model_joint.pt')
    # 'last' checkpoints hold the most recent training state (per-batch, for
    # --resume and crash recovery); model.pt/model_joint.pt hold the BEST
    # epoch, restored once at the end for downstream inference.  They diverge
    # only when the best epoch != the final epoch — resuming must continue
    # from the last epoch, not the best.
    seq_last_path = seq_path.replace('.pt', '_last.pt')
    joint_last_path = joint_path.replace('.pt', '_last.pt')
    seq_opt_path = seq_path.replace('.pt', '_opt.pt')
    joint_opt_path = joint_path.replace('.pt', '_opt.pt')
    seq_loss_path = seq_path.replace('.pt', '_loss.pkl')
    joint_loss_path = joint_path.replace('.pt', '_loss.pkl')
    meta_path = checkpoint_path(args.output_dir, 'train_meta.pkl')
    _save_model_kwargs(args.output_dir, 'model.pt', n_genes, cfg, None)
    _save_model_kwargs(args.output_dir, 'model_joint.pt', n_genes, cfg, n_lakes)

    # ------------------------------------------------------------------
    # Persistent optimizers (state survives across batches & epochs)
    # ------------------------------------------------------------------
    seq_opt = torch.optim.Adam(seq_model.parameters(), lr=args.lr,
                                weight_decay=1e-4)
    joint_opt = torch.optim.Adam(joint_model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)

    # AMP: bf16 autocast (same range as fp32 → no GradScaler needed).
    use_amp = args.device != 'cpu'
    if use_amp:
        print("[batched] Using AMP bf16 autocast")

    # Best-epoch tracking: keep the checkpoint with the lowest mean-epoch
    # criterion = edge_loss + 0.1 * commit_loss (reconstruction quality with a
    # light VQ/codebook signal) rather than the final epoch.
    def _snapshot(m):
        return {k: v.detach().cpu().clone()
                for k, v in m.state_dict().items()}

    best = {
        'seq':   {'state': None, 'crit': float('inf'), 'edge': None,
                  'commit': None, 'epoch': -1},
        'joint': {'state': None, 'crit': float('inf'), 'edge': None,
                  'commit': None, 'epoch': -1},
    }

    # ------------------------------------------------------------------
    # Resume from existing checkpoints (--resume): load the last training
    # state + optimizer momentum so training continues instead of restarting,
    # and seed best-epoch tracking from the prior best so a non-improving
    # resume never regresses model.pt.
    # ------------------------------------------------------------------
    start_epoch = 0
    if getattr(args, 'resume', False):
        for which, model, mpath_best, mpath_last, opt, opath, lp in (
                ('seq', seq_model, seq_path, seq_last_path,
                 seq_opt, seq_opt_path, seq_loss_path),
                ('joint', joint_model, joint_path, joint_last_path,
                 joint_opt, joint_opt_path, joint_loss_path)):

            # 1. Seed best-epoch tracking from the prior BEST checkpoint so a
            #    resume that fails to improve preserves the old best epoch.
            if exists(mpath_best) and exists(lp):
                # *_loss.pkl may be saved by pickle.dump (batched path) or
                # torch.save (non-batched path, e.g. VQGNN.train_joint).
                # Try pickle first, fall back to torch.load.
                try:
                    d = load_pickle(lp)
                except (pickle.UnpicklingError, AttributeError):
                    d = torch.load(lp, map_location='cpu', weights_only=False)
                e = d.get('edge_loss')
                c = d.get('commit_loss')
                if e is not None and c is not None:
                    model.load_state_dict(torch.load(
                        mpath_best, map_location=device, weights_only=True))
                    best[which] = {
                        'state': _snapshot(model),
                        'crit': e + 0.1 * c,
                        'edge': e, 'commit': c,
                        'epoch': d.get('best_epoch', 0),
                    }

            # 2. Load the LAST training state (+ optimizer) to continue from.
            #    Prefer the *_last.pt checkpoint; fall back to model.pt for
            #    legacy runs that predate this split.
            resume_from = mpath_last if exists(mpath_last) else mpath_best
            if exists(resume_from):
                model.load_state_dict(torch.load(
                    resume_from, map_location=device, weights_only=True))
                if exists(opath):
                    opt.load_state_dict(torch.load(
                        opath, map_location=device, weights_only=True))
                print(f"[batched] Resumed {which} model from {resume_from}")
            elif best[which]['state'] is not None:
                print(f"[batched] --resume: no last checkpoint for {which}; "
                      f"continuing from best epoch {best[which]['epoch']}")
            else:
                print(f"[batched] --resume: no {which} checkpoint found; "
                      f"training {which} from scratch")

        # How many epochs were already completed?
        if getattr(args, 'start_epoch', None) is not None:
            start_epoch = int(args.start_epoch)
        elif exists(meta_path):
            start_epoch = int(
                load_pickle(meta_path).get('completed_epochs', 0))
        else:
            # Legacy checkpoint without train_meta.pkl: best_epoch is a lower
            # bound on completed epochs.  Pass --start-epoch to be exact.
            fb = 0
            for lp in (seq_loss_path, joint_loss_path):
                if exists(lp):
                    try:
                        fb = max(fb, int(load_pickle(lp).get('best_epoch', 0)))
                    except (pickle.UnpicklingError, AttributeError):
                        d = torch.load(lp, map_location='cpu',
                                       weights_only=False)
                        fb = max(fb, int(d.get('best_epoch', 0)))
            # If no *_last.pt exists, the model was trained non-batched
            # (step_train_model / VQGNN.train_joint), which always completes
            # all requested epochs.  best_epoch is the *best* epoch, not the
            # total count — so default to args.epochs (the same default the
            # non-batched path used).
            if not (exists(seq_last_path) or exists(joint_last_path)):
                start_epoch = args.epochs
            else:
                start_epoch = fb
                print(f"[batched] --resume: no train_meta.pkl; using best_epoch="
                      f"{start_epoch} as lower bound "
                      f"(pass --start-epoch N to override)")

        if start_epoch >= args.epochs:
            print(f"[batched] Already completed {start_epoch} epochs "
                  f"(>= --epochs {args.epochs}); nothing to train.")
            # The models currently hold the LAST training state (loaded from
            # *_last.pt for resume), but model.pt / model_joint.pt hold the
            # BEST epoch.  Downstream steps run on whatever we return, so load
            # the best checkpoints here — otherwise a no-op resume would send
            # last-epoch weights downstream that differ from model.pt on disk.
            for m, path in ((seq_model, seq_path), (joint_model, joint_path)):
                if exists(path):
                    m.load_state_dict(torch.load(
                        path, map_location=device, weights_only=True))
            return seq_model, joint_model
        print(f"[batched] Resuming: training epochs {start_epoch + 1}.."
              f"{args.epochs} (continuing from {start_epoch} completed)")

    # ------------------------------------------------------------------
    # Training loop: epochs → batches → graphs
    # ------------------------------------------------------------------
    print(f"[batched] {n_total} graphs, {batch_size} per batch, "
          f"{args.epochs} epochs, lr={args.lr}, commit_alpha={args.commit_alpha}")

    # ------------------------------------------------------------------
    # Prefetch worker: lives across all epochs so the last batch of
    # epoch N can preload the first batch of epoch N+1.  This eliminates
    # the cold-start stall at epoch boundaries (previously ~30-60 s per
    # boundary while the GPU waited for the first batch to load).
    # ------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor

    def _load_batch(batch_indices):
        """Load one batch of graphs from disk (CPU-bound, runs in thread)."""
        list_of_edge_indices = []
        list_of_target_adjs = []
        radii_list_joint = []
        lake_ids_list = []
        recon_weights_joint = []
        seq_data = []
        for g_idx in batch_indices:
            key = all_keys[g_idx]
            graph_entry = graphs[key]
            payload = torch.load(os.fspath(graph_entry),
                                 map_location='cpu',
                                 weights_only=False)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"Expected dict bundle for {key}, got {type(payload)}")
            if payload.get('degenerate'):
                continue
            if 'graphs' not in payload:
                raise KeyError(
                    f"Bundle for {key} missing 'graphs' key — "
                    f"not a degenerate sentinel (has {sorted(payload.keys())})")
            graph_list = payload['graphs']
            if not graph_list:
                continue
            bundled_r = payload.get('radii')

            r = _resolve_graph_radii(key, radii, bundled_r,
                                     graph_list[0].num_nodes)
            # ORIGINAL (uncomment if _resolve_graph_radii breaks):
            # r = radii.get(str(key))
            # if r is None and bundled_r is not None:
            #     if isinstance(bundled_r, torch.Tensor):
            #         r = bundled_r.detach().cpu().numpy()
            #     else:
            #         r = bundled_r
            # if r is None:
            #     r = np.ones(g.num_nodes, dtype=np.float32)
            # elif isinstance(r, (str, os.PathLike)):
            #     r = np.load(os.fspath(r))
            # elif isinstance(r, torch.Tensor):
            #     r = r.detach().cpu().numpy()
            # r = np.asarray(r, dtype=np.float32)
            scaled = (r * float(args.noise_scale)
                      if args.noise_scale != 0 else r.copy())

            # Train on every reconstruction level; level i (0-based) gets
            # reconstruction loss divided by (i+1) so denser levels don't
            # dominate (their MSE is naturally larger).
            lake_name = str(key).split(' (')[0]
            for i, g in enumerate(graph_list):
                list_of_edge_indices.append(g.edge_index)
                list_of_target_adjs.append(g.target_adj)
                radii_list_joint.append(
                    r * float(args.noise_scale) if args.noise_scale != 0 else r.copy())
                recon_weights_joint.append(1.0 / (i + 1))
                if lake_name_to_id is not None:
                    lake_ids_list.append(lake_name_to_id[lake_name])
                seq_data.append((g.edge_index, g.target_adj, scaled,
                                 1.0 / (i + 1)))

        return (list_of_edge_indices, list_of_target_adjs,
                radii_list_joint, lake_ids_list, recon_weights_joint,
                seq_data)

    prefetch = ThreadPoolExecutor(max_workers=1)

    # Prime: start loading the first batch of the first epoch
    first_perm = np.random.default_rng(args.seed + start_epoch).permutation(
        n_total)
    pending = prefetch.submit(_load_batch,
                              first_perm[0:min(batch_size, n_total)])

    for ep in tqdm(range(start_epoch, args.epochs), desc="Epochs", unit="epoch"):
        # Use pre-computed perm for the first epoch (matches the prime);
        # compute fresh for subsequent epochs.
        if ep == start_epoch:
            perm = first_perm
        else:
            perm = np.random.default_rng(args.seed + ep).permutation(n_total)

        # Permutation for NEXT epoch — computed now so the last batch of
        # this epoch can prefetch the first batch of the next epoch.
        next_perm = None
        if ep < args.epochs - 1:
            next_perm = np.random.default_rng(args.seed + ep + 1).permutation(
                n_total)

        n_batches = (n_total + batch_size - 1) // batch_size
        pbar = tqdm(range(n_batches), desc=f"  Epoch {ep+1}/{args.epochs}",
                    unit="batch")

        ep_seq_edge = []
        ep_seq_commit = []
        ep_joint_edge = []
        ep_joint_commit = []

        for bi in pbar:
            # Wait for this batch to finish loading
            (list_of_edge_indices, list_of_target_adjs,
             radii_list_joint, lake_ids_list, recon_weights_joint,
             seq_data) = pending.result()

            # Submit the next batch while GPU trains this one.
            # Within-epoch: next batch of the current permutation.
            # Cross-epoch: first batch of the NEXT epoch's permutation.
            if bi < n_batches - 1:
                next_start = (bi + 1) * batch_size
                next_end = min(next_start + batch_size, n_total)
                pending = prefetch.submit(_load_batch,
                                          perm[next_start:next_end])
            elif next_perm is not None:
                pending = prefetch.submit(
                    _load_batch,
                    next_perm[0:min(batch_size, n_total)])

            n_loaded = len(list_of_edge_indices)
            if n_loaded == 0:
                del seq_data
                continue

            # ---- Train sequential on batch --------------------------
            seq_model.train()
            batch_seq_edge = []
            batch_seq_commit = []
            for edge_index, target_adj, scaled_radii, recon_weight in seq_data:
                r_tensor = (
                    torch.tensor(scaled_radii, dtype=torch.float32,
                                 device=device)
                    if args.noise_scale != 0 else None)
                edge_index_gpu = edge_index.to(device)

                # bf16 autocast: same range as fp32 → no GradScaler needed.
                with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=use_amp):
                    _, decoded, _, _, commit_loss = seq_model.forward(
                        edge_index_gpu, radii=r_tensor)
                    edge_loss = seq_model.reconstruction_loss(
                        decoded, target_adj, batch_size=recon_batch_size)
                # Multi-scale rebalancing: reconstruction level i (0-based)
                # gets its naturally larger MSE divided by (i+1).
                loss = recon_weight * edge_loss + args.commit_alpha * commit_loss

                seq_opt.zero_grad()
                loss.backward()
                seq_opt.step()

                batch_seq_edge.append(edge_loss.item())
                batch_seq_commit.append(commit_loss.item())

            # ---- Train joint on batch -------------------------------
            joint_model.train()
            batch_joint_edge = []
            batch_joint_commit = []
            for i in range(n_loaded):
                edge_index_gpu = list_of_edge_indices[i].to(device)
                target_adj = list_of_target_adjs[i]
                lake_idx = (
                    lake_ids_list[i] if lake_ids_list else i)

                r_tensor = None
                # --noise-scale 0 must DISABLE joint noise too: without this
                # check, the unscaled radii were still injected (the sequential
                # path already guarded on noise_scale != 0, the joint path did
                # not).
                if radii_list_joint and args.noise_scale != 0:
                    r_tensor = torch.tensor(
                        radii_list_joint[i], dtype=torch.float32,
                        device=device)

                # bf16 autocast: same range as fp32 → no GradScaler needed.
                with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=use_amp):
                    _, decoded, _, _, commit_loss = joint_model.forward(
                        edge_index_gpu, radii=r_tensor, lake_idx=lake_idx)
                    edge_loss = joint_model.reconstruction_loss(
                        decoded, target_adj, batch_size=recon_batch_size)
                # Multi-scale rebalancing: reconstruction level i (0-based)
                # gets its naturally larger MSE divided by (i+1).
                loss = recon_weights_joint[i] * edge_loss + \
                    args.commit_alpha * commit_loss

                joint_opt.zero_grad()
                loss.backward()
                joint_opt.step()

                batch_joint_edge.append(edge_loss.item())
                batch_joint_commit.append(commit_loss.item())

            ep_seq_edge.extend(batch_seq_edge)
            ep_seq_commit.extend(batch_seq_commit)
            ep_joint_edge.extend(batch_joint_edge)
            ep_joint_commit.extend(batch_joint_commit)

            pbar.set_postfix(
                seq=f"{np.mean(batch_seq_edge):.4f}" if batch_seq_edge else "N/A",
                joint=f"{np.mean(batch_joint_edge):.4f}" if batch_joint_edge else "N/A",
            )

            # Free batch memory: dropping references lets the caching
            # allocator reuse these blocks for the next batch. No
            # empty_cache() — it would force a sync + allocator thrash.
            del (list_of_edge_indices, list_of_target_adjs,
                 radii_list_joint, lake_ids_list, recon_weights_joint,
                 seq_data)

            # Save 'last' checkpoint (crash-recovery + --resume): model +
            # optimizer together so a resumed run continues with consistent
            # Adam momentum.  model.pt (best epoch) is written once at the end.
            torch.save(seq_model.state_dict(), seq_last_path)
            torch.save(joint_model.state_dict(), joint_last_path)
            torch.save(seq_opt.state_dict(), seq_opt_path)
            torch.save(joint_opt.state_dict(), joint_opt_path)

        # ---- End-of-epoch summary + best-epoch tracking ---------------
        # Criterion = mean(edge_loss) + 0.1 * mean(commit_loss).
        if ep_seq_edge:
            seq_e = float(np.mean(ep_seq_edge))
            seq_c = float(np.mean(ep_seq_commit))
            seq_crit = seq_e + 0.1 * seq_c
            if seq_crit < best['seq']['crit']:
                best['seq'] = {'state': _snapshot(seq_model), 'crit': seq_crit,
                               'edge': seq_e, 'commit': seq_c, 'epoch': ep + 1}
        if ep_joint_edge:
            joint_e = float(np.mean(ep_joint_edge))
            joint_c = float(np.mean(ep_joint_commit))
            joint_crit = joint_e + 0.1 * joint_c
            if joint_crit < best['joint']['crit']:
                best['joint'] = {'state': _snapshot(joint_model),
                                 'crit': joint_crit, 'edge': joint_e,
                                 'commit': joint_c, 'epoch': ep + 1}
        seq_edge_str = (f"edge={np.mean(ep_seq_edge):.4f}") if ep_seq_edge else "N/A"
        seq_commit_str = (f"commit={np.mean(ep_seq_commit):.4f}") if ep_seq_commit else "N/A"
        joint_edge_str = (f"edge={np.mean(ep_joint_edge):.4f}") if ep_joint_edge else "N/A"
        joint_commit_str = (f"commit={np.mean(ep_joint_commit):.4f}") if ep_joint_commit else "N/A"
        seq_best = (f" [best ep {best['seq']['epoch']} "
                    f"{best['seq']['crit']:.4f}]") if best['seq']['state'] else ""
        joint_best = (f" [best ep {best['joint']['epoch']} "
                      f"{best['joint']['crit']:.4f}]") if best['joint']['state'] else ""
        print(f"  Epoch {ep+1}/{args.epochs} | "
              f"seq {seq_edge_str} {seq_commit_str}{seq_best} | "
              f"joint {joint_edge_str} {joint_commit_str}{joint_best}")

        # Record completed epochs so a future --resume knows where to pick up.
        save_pickle({'completed_epochs': ep + 1}, meta_path)

    prefetch.shutdown(wait=True)

    # Restore the best epoch into both models and save it as the checkpoint, so
    # downstream steps and the on-disk weights reflect the best epoch rather
    # than the final one.
    import pickle as _pk
    for which, model, path in (('seq', seq_model, seq_path),
                               ('joint', joint_model, joint_path)):
        b = best[which]
        if b['state'] is None:
            continue
        model.load_state_dict(b['state'])
        torch.save(model.state_dict(), path)
        loss_path = path.replace('.pt', '_loss.pkl')
        _pk.dump({'edge_loss': b['edge'], 'commit_loss': b['commit'],
                  'best_epoch': b['epoch']},
                 open(loss_path, 'wb'))
        print(f"[batched] {which}: best epoch {b['epoch']} "
              f"(edge_loss={b['edge']:.4f}, commit_loss={b['commit']:.4f}) "
              f"-> {path}")

    print(f"[batched] Models saved to {seq_path} and {joint_path}")
    return seq_model, joint_model


def step_train_model(args, cfg, graphs, radii, preloaded=None):
    from surge.vqgnn import VQGNN
    import torch

    joint = args.train_joint
    model_filename = 'model_joint.pt' if joint else 'model.pt'
    model_path = checkpoint_path(args.output_dir, model_filename)

    n_genes = cfg['n_genes']

    # Resolve unique lakes for joint training
    n_lakes = None
    lake_name_to_id = None
    if joint:
        lake_names = sorted(set(
            str(k).split(' (')[0] for k in graphs.keys()
        ))
        n_lakes = len(lake_names)
        lake_name_to_id = {name: i for i, name in enumerate(lake_names)}
        print(f"[Step 3] Joint training: {n_lakes} unique lakes, "
              f"{len(graphs)} graph keys")

    if not args.force and exists(model_path):
        print(f"[Step 3] Loading cached {'joint ' if joint else ''}model...")
        model = _build_vqgnn(cfg, n_genes, n_lakes=n_lakes)
        # ORIGINAL (uncomment if _build_vqgnn breaks):
        # model = VQGNN(
        #     n_nodes=n_genes,
        #     in_channels=cfg.get('in_channels', 64),
        #     hidden_channels=cfg.get('hidden_channels', 64),
        #     out_channels=cfg.get('out_channels', 16),
        #     num_layers=cfg.get('num_layers', 3),
        #     dropout=cfg.get('dropout', 0.3),
        #     codebook_channels=cfg.get('codebook_channels', 16),
        #     codebook_size=cfg.get('codebook_size', 100),
        #     decoder_channels=cfg.get('decoder_channels', 256),
        #     n_lakes=n_lakes,
        # )
        model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=True)
        )
        try:
            model.to(args.device)
        except RuntimeError:
            print(f"[Step 3] CUDA not available, using CPU")
            args.device = 'cpu'
            model.to('cpu')
        print(f"[Step 3] Loaded cached model, device: "
              f"{next(model.parameters()).device}")
        _save_model_kwargs(args.output_dir, model_filename, n_genes, cfg, n_lakes)
        return model

    print(f"[Step 3] Training VQGNN model{' (joint)' if joint else ''}...")

    model = _build_vqgnn(cfg, n_genes, n_lakes=n_lakes)
    # ORIGINAL (uncomment if _build_vqgnn breaks):
    # model = VQGNN(
    #     n_nodes=n_genes,
    #     in_channels=cfg.get('in_channels', 64),
    #     hidden_channels=cfg.get('hidden_channels', 64),
    #     out_channels=cfg.get('out_channels', 16),
    #     num_layers=cfg.get('num_layers', 3),
    #     dropout=cfg.get('dropout', 0.3),
    #     codebook_channels=cfg.get('codebook_channels', 16),
    #     codebook_size=cfg.get('codebook_size', 100),
    #     decoder_channels=cfg.get('decoder_channels', 256),
    #     n_lakes=n_lakes,
    # )
    model.to(args.device)
    _save_model_kwargs(args.output_dir, model_filename, n_genes, cfg, n_lakes)
    print(f"[Step 3] Model device: {next(model.parameters()).device}, "
          f"args.device: {args.device}")

    if joint:
        if preloaded is not None:
            (list_of_edge_indices, list_of_target_adjs, radii_list,
             lake_ids_list, recon_weights_list) = preloaded['joint']
            n_graphs = preloaded['n_loaded']
            print(f"  Using pre-loaded data: {n_graphs} graphs "
                  f"({len(lake_names)} unique lakes)")
        else:
            # --- Build flat lists from disk ---
            list_of_edge_indices = []
            list_of_target_adjs = []
            radii_list = []
            lake_ids_list = []
            recon_weights_list = []

            keys_sorted = sorted(graphs.keys())
            n_total = len(keys_sorted)
            skipped_degenerate = []
            print(f"[Step 3] Loading {n_total} graph bundles from disk...")
            for key in tqdm(keys_sorted, desc="  Loading graphs", unit="key"):
                graph_entry = graphs[key]
                bundled_r = None
                graph_list = None

                if isinstance(graph_entry, (str, os.PathLike)):
                    payload = torch.load(os.fspath(graph_entry),
                                         map_location='cpu',
                                         weights_only=False)
                    if not isinstance(payload, dict):
                        raise TypeError(
                            f"Expected dict bundle for {key}, got {type(payload)}")
                    # Degenerate sentinel: saved without 'graphs' when the
                    # reconstruction was all-empty or all-dense (>99.9%).
                    if payload.get('degenerate'):
                        skipped_degenerate.append(str(key))
                        continue
                    if 'graphs' not in payload:
                        raise KeyError(
                            f"Bundle for {key} missing 'graphs' key — "
                            f"not a degenerate sentinel (has {sorted(payload.keys())})")
                    graph_list = payload['graphs']
                    bundled_r = payload.get('radii')
                elif isinstance(graph_entry, (list, tuple)):
                    graph_list = list(graph_entry)
                elif isinstance(graph_entry, dict) and graph_entry.get('degenerate'):
                    skipped_degenerate.append(str(key))
                    continue
                else:
                    graph_list = [graph_entry]

                if not graph_list:
                    continue

                r = _resolve_graph_radii(key, radii, bundled_r,
                                         graph_list[0].num_nodes)
                # ORIGINAL (uncomment if _resolve_graph_radii breaks):
                # r = radii.get(str(key))
                # if r is None and bundled_r is not None:
                #     if isinstance(bundled_r, torch.Tensor):
                #         r = bundled_r.detach().cpu().numpy()
                #     else:
                #         r = bundled_r
                # if r is None:
                #     r = np.ones(graph.num_nodes, dtype=np.float32)
                # elif isinstance(r, (str, os.PathLike)):
                #     r = np.load(os.fspath(r))
                # elif isinstance(r, torch.Tensor):
                #     r = r.detach().cpu().numpy()
                # r = np.asarray(r, dtype=np.float32)
                if args.noise_scale != 0:
                    r = r * float(args.noise_scale)

                # Train on EVERY reconstruction level; level i (0-based)
                # gets its reconstruction loss divided by (i+1) so denser
                # levels don't dominate.
                lake_name = str(key).split(' (')[0]
                for i, graph in enumerate(graph_list):
                    list_of_edge_indices.append(graph.edge_index)
                    list_of_target_adjs.append(graph.target_adj)
                    radii_list.append(r)
                    recon_weights_list.append(1.0 / (i + 1))
                    lake_ids_list.append(lake_name_to_id[lake_name])

            if skipped_degenerate:
                print(f"  Skipped {len(skipped_degenerate)} degenerate "
                      f"graph(s): {', '.join(sorted(skipped_degenerate)[:10])}"
                      f"{' ...' if len(skipped_degenerate) > 10 else ''}")
            print(f"  Training on {len(list_of_edge_indices)} graphs "
                  f"({len(lake_names)} unique lakes)")

        model.train_joint(
            model_save_path=model_path,
            list_of_edge_indices=list_of_edge_indices,
            list_of_target_adjs=list_of_target_adjs,
            epochs=args.epochs,
            lr=args.lr,
            commit_alpha=args.commit_alpha,
            # --noise-scale 0 disables noise entirely: pass radii=None so the
            # unscaled radii are not injected (they otherwise would be).
            radii=radii_list if args.noise_scale != 0 else None,
            lake_ids=lake_ids_list,
            recon_weights=recon_weights_list,
        )
    else:
        from surge.embedder import LakeEmbedder

        embedder = LakeEmbedder(model, device=args.device)
        embedder.train(
            graphs_dict=graphs,
            radii_dict=radii,
            save_path=model_path,
            epochs=args.epochs,
            lr=args.lr,
            commit_alpha=args.commit_alpha,
            noise_scale=args.noise_scale,
            recon_batch_size=cfg.get('recon_batch_size', 256),
            skip_oom=cfg.get('skip_oom', True),
            preloaded_seq=preloaded['sequential'] if preloaded else None,
        )

    print(f"  Saved model to {model_path}")
    return model


# ---------------------------------------------------------------------------
# Step 4: Generate lake embeddings
# ---------------------------------------------------------------------------

def step_generate_embeddings(args, graphs, model):
    from surge.embedder import LakeEmbedder

    embeddings_path = checkpoint_path(args.output_dir,
                                       _joint_path(args, 'embeddings.pkl'))

    if not args.force and exists(embeddings_path):
        print("[Step 4] Loading cached embeddings...")
        return load_pickle(embeddings_path)

    print("[Step 4] Generating VQ code histogram embeddings...")
    embedder = LakeEmbedder(model, device=args.device)
    # A jointly-trained model must be conditioned on the same lake indices
    # used at training time, or its learned per-lake shift is silently
    # dropped.  Build the mapping from the same sorted-lake-name scheme that
    # the training path uses so the indices line up.
    lake_name_to_id = {
        name: i for i, name in enumerate(
            sorted(set(str(k).split(' (')[0] for k in graphs.keys())))
    } if getattr(model, 'lake_emb', None) is not None else None
    embeddings = embedder.embed_all(graphs, lake_name_to_id=lake_name_to_id)

    save_pickle(embeddings, embeddings_path)
    print(f"  Saved {len(embeddings)} embeddings to {embeddings_path}")
    return embeddings


# ---------------------------------------------------------------------------
# Step 5: Build gene-VQ mappings
# ---------------------------------------------------------------------------

def step_build_gene_mappings(args, graphs, model, sd):
    from surge.embedder import LakeEmbedder

    mappings_path = checkpoint_path(args.output_dir,
                                     _joint_path(args, 'gene_mappings.pkl'))

    if not args.force and exists(mappings_path):
        print("[Step 5] Loading cached gene-VQ mappings...")
        return load_pickle(mappings_path)

    print("[Step 5] Building gene-VQ code mappings...")
    embedder = LakeEmbedder(model, device=args.device)
    lake_name_to_id = {
        name: i for i, name in enumerate(
            sorted(set(str(k).split(' (')[0] for k in graphs.keys())))
    } if getattr(model, 'lake_emb', None) is not None else None
    vq_to_gene, gene_to_vq = embedder.build_gene_vq_mappings(
        graphs, lake_name_to_id=lake_name_to_id)

    gene_names = sd.gene_names if hasattr(sd, 'gene_names') else None
    mappings = {
        'vq_to_gene': vq_to_gene,
        'gene_to_vq': gene_to_vq,
        'gene_names': gene_names,
    }

    save_pickle(mappings, mappings_path)
    print(f"  Saved mappings to {mappings_path}")
    return mappings


# ---------------------------------------------------------------------------
# Step 6: Temporal slope analysis
# ---------------------------------------------------------------------------

def step_temporal_slopes(args, embeddings, sd):
    from surge.analysis import LakeAnalyzer

    slopes_path = checkpoint_path(args.output_dir,
                                   _joint_path(args, 'slopes.pkl'))

    if not args.force and exists(slopes_path):
        print("[Step 6] Loading cached slope analysis...")
        return load_pickle(slopes_path)

    print("[Step 6] Computing temporal slopes (Source vs Recipient)...")
    analyzer = LakeAnalyzer(embeddings, data=sd)

    # Pooled across ALL stratifications (legacy, also used by wasserstein_grid.png)
    wass_all = analyzer.wasserstein_temporal(base_year=args.base_year)
    wass_all, p95 = LakeAnalyzer.scale_wasserstein_p95(wass_all)
    result = analyzer.source_vs_recipient_slope_test(wass_all)
    result['wasserstein_p95'] = p95

    # Per-stratification (used by wasserstein_grid_{strat}.png)
    per_strat = {}
    for strat in ['year_lake', 'sex_year_lake', 'infection_year_lake']:
        sub = analyzer.for_stratification(strat)
        # Skip suffixed stratifications that have no unambiguous per-lake
        # temporal series (sex/infection), loudly.
        wass = sub.wasserstein_temporal_or_skip(base_year=args.base_year)
        if not wass or len(wass) < 3:
            per_strat[strat] = {'error': 'too few lakes with temporal data'}
            continue
        wass_s, p95_s = LakeAnalyzer.scale_wasserstein_p95(wass)
        sr = sub.source_vs_recipient_slope_test(wass)
        # Keep only JSON-serializable scalars
        entry = {k: float(v) if v is not None else None
                 for k, v in [('wasserstein_p95', p95_s)]}
        for k in ('ttest_stat', 'ttest_pvalue',
                  'mannwhitney_stat', 'mannwhitney_pvalue'):
            entry[k] = float(sr[k]) if sr.get(k) is not None else None
        src = sr.get('source_slopes', [])
        rec = sr.get('recipient_slopes', [])
        slopes = sr.get('slopes', {})
        entry['n_lakes'] = len(slopes)
        entry['mean_slope'] = float(np.mean(list(slopes.values()))) if slopes else None
        entry['source_n'] = len(src)
        entry['source_mean_slope'] = float(np.mean(src)) if src else None
        entry['recipient_n'] = len(rec)
        entry['recipient_mean_slope'] = float(np.mean(rec)) if rec else None
        per_strat[strat] = entry
    result['stratifications'] = per_strat

    save_pickle(result, slopes_path)

    print(f"  Wasserstein P95 (pooled): {p95:.6f}")
    print(f"  Source lakes: {len(result.get('source_slopes', []))}")
    print(f"  Recipient lakes: {len(result.get('recipient_slopes', []))}")
    if result.get('ttest_pvalue') is not None:
        print(f"  t-test p-value (pooled): {result['ttest_pvalue']:.4f}")
        print(f"  Mann-Whitney p-value (pooled): {result['mannwhitney_pvalue']:.4f}")
    for strat, entry in per_strat.items():
        if entry.get('ttest_pvalue') is not None:
            print(f"  [{strat}] t-test p={entry['ttest_pvalue']:.4f}, "
                  f"MW p={entry['mannwhitney_pvalue']:.4f}, "
                  f"src n={entry.get('source_n',0)}, "
                  f"rec n={entry.get('recipient_n',0)}")
        else:
            print(f"  [{strat}] {entry.get('error', 'no test')}")
    print(f"  Saved to {slopes_path}")
    return result


# ---------------------------------------------------------------------------
# Step 7: Hierarchical clustering
# ---------------------------------------------------------------------------

def step_hierarchical_clustering(args, embeddings, sd):
    from surge.analysis import LakeAnalyzer

    clust_path = checkpoint_path(args.output_dir,
                                  _joint_path(args, 'clustering.pkl'))

    if not args.force and exists(clust_path):
        print("[Step 7] Loading cached clustering...")
        return load_pickle(clust_path)

    print("[Step 7] Running hierarchical clustering...")
    analyzer = LakeAnalyzer(embeddings, data=sd)
    result = analyzer.hierarchical_clustering(
        method=args.linkage_method,
        n_clusters=args.n_clusters,
    )

    save_pickle(result, clust_path)
    print(f"  Saved to {clust_path}")
    return result


# ---------------------------------------------------------------------------
# Step 8: Label-permutation test (Source/Recipient)
# ---------------------------------------------------------------------------

def step_label_permutation(args, embeddings, sd):
    """Run label-permutation tests on biologically meaningful groupings.

    Runs three tests:
      1. Source lakes only — ecotype (Benthic vs Limnetic)
      2. Recipient lakes only — ancestry (BenthicPool vs LimneticPool)
      3. Recipient lakes only — habitat ecotype (Benthic vs Limnetic)

    Each test shuffles labels to build a null distribution and computes
    an empirical p-value for the observed silhouette score.
    """
    from surge.analysis import LakeAnalyzer

    perm_path = checkpoint_path(args.output_dir,
                                 _joint_path(args, 'label_perm.pkl'))

    if not args.force and exists(perm_path):
        print("[Step 8] Loading cached label-permutation results...")
        return load_pickle(perm_path)

    print(f"[Step 8] Label-permutation tests "
          f"(n={args.label_permutations})...")
    analyzer = LakeAnalyzer(embeddings, data=sd)

    results = {}

    # --- Test 1: Source lakes, ecotype (Benthic vs Limnetic) ---
    src = analyzer.for_role('Source')
    if len(src.keys) >= 4:
        print(f"\n  Source lakes only — ecotype (n={len(src.keys)} keys)")
        results['source_ecotype'] = src.label_permutation_test(
            'ecotype', n_permutations=args.label_permutations)
        r = results['source_ecotype']
        print(f"    Observed silhouette: {r['observed']:.4f}")
        print(f"    Permutation p-value: {r['p_value']:.4f}")
    else:
        print(f"  Source lakes: only {len(src.keys)} keys — skipping")

    # --- Test 2: Recipient lakes, ancestry (BenthicPool vs LimneticPool) ---
    rec = analyzer.for_role('Recipient')
    rec_anc = rec.for_subset([
        k for k in rec.keys
        if rec.build_labels('genotype_benthic_limnetic').get(k) is not None
    ])
    if len(rec_anc.keys) >= 4:
        print(f"\n  Recipient lakes only — ancestry "
              f"(n={len(rec_anc.keys)} keys)")
        results['recipient_ancestry'] = rec_anc.label_permutation_test(
            'genotype_benthic_limnetic',
            n_permutations=args.label_permutations)
        r = results['recipient_ancestry']
        print(f"    Observed silhouette: {r['observed']:.4f}")
        print(f"    Permutation p-value: {r['p_value']:.4f}")
    else:
        print(f"  Recipient ancestry: only {len(rec_anc.keys)} keys "
              f"— skipping")

    # --- Test 3: Recipient lakes, habitat ecotype (Benthic vs Limnetic) ---
    rec_eco = rec.for_subset([
        k for k in rec.keys
        if rec.build_labels('ecotype').get(k) is not None
    ])
    if len(rec_eco.keys) >= 4:
        print(f"\n  Recipient lakes only — habitat ecotype "
              f"(n={len(rec_eco.keys)} keys)")
        results['recipient_ecotype'] = rec_eco.label_permutation_test(
            'ecotype', n_permutations=args.label_permutations)
        r = results['recipient_ecotype']
        print(f"    Observed silhouette: {r['observed']:.4f}")
        print(f"    Permutation p-value: {r['p_value']:.4f}")
    else:
        print(f"  Recipient ecotype: only {len(rec_eco.keys)} keys "
              f"— skipping")

    save_pickle(results, perm_path)
    print(f"\n  Saved {len(results)} tests to {perm_path}")
    return results


# ---------------------------------------------------------------------------
# Step 9: Generate figures
# ---------------------------------------------------------------------------

def step_generate_figures(args, embeddings, sd, mappings, slopes, clustering,
                          label_perm, model=None):
    from surge.analysis import LakeAnalyzer
    from surge.plotting import LakePlotter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if args.train_joint:
        figures_dir = os.path.join(args.output_dir, 'figures_joint')
    else:
        figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    analyzer = LakeAnalyzer(embeddings, data=sd)
    strat_types = ['year_lake', 'sex_year_lake', 'infection_year_lake']

    print("[Step 9] Generating figures...")

    # ---- Cross-stratification: Wasserstein temporal grid ----
    wass = analyzer.wasserstein_temporal(base_year=args.base_year)
    if wass:
        wass_scaled, _ = LakeAnalyzer.scale_wasserstein_p95(wass)
        slope_result = analyzer.source_vs_recipient_slope_test(wass)
        t_p = slope_result.get('ttest_pvalue')
        mw_p = slope_result.get('mannwhitney_pvalue')
        if t_p is not None and mw_p is not None:
            sup = (f"Wasserstein Temporal Drift — "
                   f"Source vs Recipient: t-test p={t_p:.4f}, "
                   f"Mann-Whitney p={mw_p:.4f}")
        else:
            sup = "Wasserstein Temporal Drift"
        fig = LakePlotter.wasserstein_grid(
            wass_scaled, data=sd,
            ylabel='Wasserstein Distance',
            suptitle=sup,
        )
        fig.savefig(os.path.join(figures_dir, 'wasserstein_grid.png'), dpi=300)
        plt.close(fig)
        print("  Saved wasserstein_grid.png")

    # ---- Lake embedding PCA (joint training only) ----
    if args.train_joint and model is not None:
        print("  Generating lake embedding PCA...")
        try:
            lake_emb_weight = model.lake_emb.weight.detach().cpu().numpy()
            lake_name_list = sorted(set(
                str(k).split(' (')[0] for k in embeddings.keys()
            ))
            fig, ax = plt.subplots(figsize=(12, 10))
            LakePlotter.lake_embedding_pca(
                lake_name_list, lake_emb_weight, data=sd, ax=ax)
            fig.savefig(os.path.join(figures_dir, 'lake_embeddings.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  Saved lake_embeddings.png")
        except Exception as e:
            print(f"  Lake embedding PCA skipped: {e}")

    # ---- Silhouette scan: optimal k ----
    print("  Generating silhouette scan...")
    sil_scores = {}
    for strat in strat_types:
        sub = analyzer.for_stratification(strat)
        if len(sub.keys) >= 3:
            sil_scores[strat] = sub.silhouette_scan(max_k=min(10, len(sub.keys) - 1))
    if sil_scores:
        fig, ax = plt.subplots(figsize=(8, 5))
        LakePlotter.silhouette_scan(sil_scores, ax=ax)
        fig.savefig(os.path.join(figures_dir, 'silhouette_scan.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        for strat, scores in sil_scores.items():
            best_k = max(scores, key=lambda k: scores[k])
            print(f"    {strat}: best k={best_k} "
                  f"(silhouette={scores[best_k]:.4f})")
        print("  Saved silhouette_scan.png")

    # ---- Silhouette: Source lakes (ecotype labels) ----
    src_analyzer = analyzer.for_role('Source')
    if len(src_analyzer.keys) >= 4:
        print("\n  Source-lake ecotype silhouette...")
        for strat in strat_types:
            sub = src_analyzer.for_stratification(strat)
            labels = sub.build_labels('ecotype')
            if len(set(labels.values())) >= 2:
                sil = sub.silhouette(labels)
                print(f"    {strat}: ecotype silhouette = {sil:.4f} "
                      f"(n={len(labels)})")

    # ---- Silhouette: Recipient lakes (ancestry labels) ----
    rec_analyzer = analyzer.for_role('Recipient')
    if len(rec_analyzer.keys) >= 4:
        print("\n  Recipient-lake ancestry silhouette...")
        for strat in strat_types:
            sub = rec_analyzer.for_stratification(strat)
            labels = sub.build_labels('genotype_benthic_limnetic')
            if len(set(labels.values())) >= 2:
                sil = sub.silhouette(labels)
                print(f"    {strat}: ancestry silhouette = {sil:.4f} "
                      f"(n={len(labels)})")

    # ---- Per-stratification figures ----
    # Debug: show key distribution across stratification types
    key_groups = analyzer.keys_by_stratification()
    print(f"  Keys per stratification: " +
          " | ".join(f"{t}={len(v)}" for t, v in key_groups.items()))

    for strat in tqdm(strat_types, desc="  Per-stratification figures"):
        sub = analyzer.for_stratification(strat)
        if not sub.keys:
            print(f"  Skipping {strat} — no keys "
                  f"(available: {list(key_groups.keys())})")
            continue
        print(f"  --- {strat} ({len(sub.keys)} keys) ---")

        # -- PCA scatter --
        projected, _ = sub.pca_project(n_components=2)
        fig, ax = plt.subplots(figsize=(12, 10))
        LakePlotter.pca_scatter(projected, sub.keys, sd, color_by='genotype',
                                title=f"PCA of Lake VQ-Code Embeddings — {strat}",
                                ax=ax)
        fig.savefig(os.path.join(figures_dir, f'pca_genotype_{strat}.png'), dpi=300)
        plt.close(fig)
        print(f"    Saved pca_genotype_{strat}.png")

        # -- Wasserstein temporal grid (per-stratification) --
        sub_wass = sub.wasserstein_temporal_or_skip(base_year=args.base_year)
        if sub_wass:
            sub_wass_scaled, sub_p95 = LakeAnalyzer.scale_wasserstein_p95(sub_wass)
            n_lakes = len(sub_wass_scaled)
            # Source vs Recipient slope test for this stratification
            sub_slope = sub.source_vs_recipient_slope_test(sub_wass)
            t_p = sub_slope.get('ttest_pvalue')
            mw_p = sub_slope.get('mannwhitney_pvalue')
            if t_p is not None and mw_p is not None:
                sub_sup = (f"Wasserstein Temporal Drift — {strat}\n"
                           f"Source vs Recipient: t-test p={t_p:.4f}, "
                           f"Mann-Whitney p={mw_p:.4f}")
            else:
                n_src = len(sub_slope.get('source_slopes', []))
                n_rcp = len(sub_slope.get('recipient_slopes', []))
                sub_sup = (f"Wasserstein Temporal Drift — {strat}\n"
                           f"(insufficient data: {n_src} Source, "
                           f"{n_rcp} Recipient lakes)")
            fig = LakePlotter.wasserstein_grid(
                sub_wass_scaled, data=sd,
                ylabel='Wasserstein Distance',
                ncols=min(4, max(1, n_lakes)),
                figsize=(min(16, n_lakes * 4), max(6, n_lakes * 0.75)),
                suptitle=sub_sup,
            )
            fig.savefig(os.path.join(figures_dir, f'wasserstein_grid_{strat}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved wasserstein_grid_{strat}.png "
                  f"(P95={sub_p95:.6f}, {n_lakes} lakes)")

        # -- Dendrogram --
        sub_clust = sub.hierarchical_clustering(
            method=args.linkage_method, n_clusters=args.n_clusters,
        )
        if sub_clust and 'linkage' in sub_clust:
            from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

            # Compute cluster enrichments for annotation
            n_clust = args.n_clusters or 4
            clust_info = sub.analyze_clusters(n_clusters=n_clust)

            # --- Palette: 5 distinct colours for the 5 clusters ---
            palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                       '#ff7f00']
            set_link_color_palette(palette)

            # --- Colour threshold: pick the merge distance where we go
            #     from n_clust+1 to n_clust clusters, so at least n_clust
            #     distinct subtrees are coloured.  Using n_clust-1 (the
            #     standard formula) can produce only 2-3 coloured groups
            #     when the tree is unbalanced.
            Z = sub_clust['linkage']
            if n_clust > 1 and n_clust <= len(Z) - 1:
                # Z[-(n_clust), 2] is the merge that takes us from
                # n_clust+1 → n_clust clusters.  Multiply by 1.001 to
                # put the threshold just above this merge so exactly
                # n_clust+1 subtrees are below it, guaranteeing ≥4
                # coloured groups.
                color_threshold = Z[-(n_clust), 2] * 1.001
            else:
                color_threshold = None

            # Extra height for cluster-annotation boxes below the dendrogram
            n_clusters_annotated = len(clust_info['clusters'])
            extra_height = max(1.5, n_clusters_annotated * 0.9)
            fig, ax = plt.subplots(
                figsize=(18, max(7, len(sub.keys) * 0.18) + extra_height),
            )
            import re as _re
            _clean_dendro_labels = [_re.sub(r'(\d{4})\.0', r'\1', str(lbl))
                                    for lbl in sub_clust['keys']]
            dendrogram(
                Z, labels=_clean_dendro_labels,
                leaf_font_size=8, ax=ax,
                color_threshold=color_threshold,
                above_threshold_color='#888888',
            )
            ax.set_title(f"Hierarchical Clustering — {strat}")

            # --- Map cluster IDs to the same palette used for dendrogram ---
            cluster_color = {}
            for cid in sorted(clust_info['clusters'].keys()):
                cluster_color[cid] = palette[(cid - 1) % len(palette)]

            # --- Build enrichment text boxes, placed BELOW the dendrogram ---
            def _stars(p):
                if p < 0.01:
                    return '**'
                elif p < 0.05:
                    return '*'
                return ''

            # Render figure once so we know the axes bounding box
            fig.canvas.draw()
            ax_bbox = ax.get_position()  # in figure coords

            # Lay out cluster boxes horizontally below the axes, all in one row
            boxes_per_row = 5
            box_width = (ax_bbox.width - 0.02) / boxes_per_row
            row_idx = 0
            col_idx = 0
            for cid in sorted(clust_info['clusters'].keys()):
                ci = clust_info['clusters'][cid]
                color = cluster_color.get(cid, '#333333')
                parts = [f"Cluster {cid} (n={ci['size']}):"]
                for combo, ed in ci.get('role_ecotype_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    parts.append(
                        f"  {combo}{s} "
                        f"({ed['count']}/{ci['size']}, "
                        f"p={ed['fisher_p']:.3f})")
                for role, ed in ci.get('role_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    parts.append(
                        f"  {role}{s} "
                        f"({ed['count']}/{ci['size']}, "
                        f"p={ed['fisher_p']:.3f})")
                for eco, ed in ci.get('ecotype_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    parts.append(
                        f"  {eco}{s} "
                        f"({ed['count']}/{ci['size']}, "
                        f"p={ed['fisher_p']:.3f})")
                for lab, ed in ci.get('sex_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    parts.append(
                        f"  {lab}{s} "
                        f"({ed['count']}/{ci['size']}, "
                        f"p={ed['fisher_p']:.3f})")
                for lab, ed in ci.get('infection_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    parts.append(
                        f"  {lab}{s} "
                        f"({ed['count']}/{ci['size']}, "
                        f"p={ed['fisher_p']:.3f})")
                if len(parts) <= 1:
                    continue

                block = '\n'.join(parts)
                # Position below the axes
                bx = ax_bbox.x0 + col_idx * box_width + box_width * 0.02
                by = ax_bbox.y0 - 0.04 - (row_idx + 1) * 0.14
                fig.text(
                    bx, by, block,
                    fontsize=8,
                    verticalalignment='top',
                    fontfamily='monospace', color=color,
                    bbox=dict(boxstyle='round', facecolor='lightyellow',
                              alpha=0.85, edgecolor=color, linewidth=1.2),
                )
                col_idx += 1
                if col_idx >= boxes_per_row:
                    col_idx = 0
                    row_idx += 1

            fig.savefig(os.path.join(figures_dir, f'dendrogram_{strat}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved dendrogram_{strat}.png")

            # Console printout — all enrichments
            for cid in sorted(clust_info['clusters'].keys()):
                ci = clust_info['clusters'][cid]
                lines = [f"    Cluster {cid}: {ci['size']} keys"]
                for combo, ed in ci.get('role_ecotype_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    lines.append(
                        f"      {combo}{s}: {ed['count']}/{ci['size']} "
                        f"({ed['pct']:.0%}) p={ed['fisher_p']:.3f}")
                for role, ed in ci.get('role_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    lines.append(
                        f"      {role}{s}: {ed['count']}/{ci['size']} "
                        f"({ed['pct']:.0%}) p={ed['fisher_p']:.3f}")
                for eco, ed in ci.get('ecotype_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    lines.append(
                        f"      {eco}{s}: {ed['count']}/{ci['size']} "
                        f"({ed['pct']:.0%}) p={ed['fisher_p']:.3f}")
                for lab, ed in ci.get('sex_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    lines.append(
                        f"      {lab}{s}: {ed['count']}/{ci['size']} "
                        f"({ed['pct']:.0%}) p={ed['fisher_p']:.3f}")
                for lab, ed in ci.get('infection_enrichment', {}).items():
                    s = _stars(ed['fisher_p'])
                    lines.append(
                        f"      {lab}{s}: {ed['count']}/{ci['size']} "
                        f"({ed['pct']:.0%}) p={ed['fisher_p']:.3f}")
                if 'year_range' in ci:
                    lines.append(f"      years: {ci['year_range']} "
                                 f"(mean {ci['year_mean']:.1f})")
                for line in lines:
                    print(line)

        # -- VQ code distribution (aggregated across all keys in this stratification) --
        if mappings:
            strat_keys_list = sorted(
                set(sub.keys) & set(mappings['vq_to_gene'].keys())
            )
            if strat_keys_list:
                # Aggregate across all lake keys so:
                #   Panel A: unique genes assigned to each VQ code (across all lakes)
                #   Panel B: distinct VQ codes each gene takes (across different lake contexts)
                from collections import defaultdict
                agg_vq_to_gene = defaultdict(set)
                agg_gene_to_vq = defaultdict(set)
                for key in strat_keys_list:
                    for code, genes in mappings['vq_to_gene'].get(key, {}).items():
                        agg_vq_to_gene[code].update(genes)
                    for gene_idx, codes in mappings['gene_to_vq'].get(key, {}).items():
                        agg_gene_to_vq[gene_idx].update(codes)
                agg_vq_to_gene = {c: list(gs) for c, gs in agg_vq_to_gene.items()}
                agg_gene_to_vq = {g: list(cs) for g, cs in agg_gene_to_vq.items()}

                # Compute diversity stats for logging
                code_counts = [len(cs) for cs in agg_gene_to_vq.values()]
                unique_codes_per_gene = len([c for c in code_counts if c > 1])
                fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                LakePlotter.code_distribution(agg_vq_to_gene, agg_gene_to_vq, ax=ax)
                fig.suptitle(f"VQ Code Distribution — {strat} "
                             f"({len(strat_keys_list)} keys, "
                             f"{unique_codes_per_gene} genes with >1 code)",
                             fontsize=12, y=1.02)
                fig.savefig(os.path.join(figures_dir, f'code_distribution_{strat}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved code_distribution_{strat}.png "
                      f"({unique_codes_per_gene}/{len(agg_gene_to_vq)} genes "
                      f"have >1 distinct VQ code)")

            # -- Infection code enrichment (infection_year_lake only) --
            if strat == 'infection_year_lake':
                code_enrich = sub.infection_code_enrichment(
                    codebook_size=args.codebook_size)
                if code_enrich:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    LakePlotter.infection_codes(code_enrich, top_n=20, ax=ax)
                    fig.savefig(os.path.join(figures_dir,
                                'infection_code_enrichment.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print("    Saved infection_code_enrichment.png")

                    # Export gene lists for GO analysis.
                    # Use raw p < 0.05 rather than FDR q < 0.05 so that
                    # low-but-real signals (sex, role) also produce gene lists
                    # for g:Profiler and WGCNA overlap.  FDR q-values are still
                    # written to the CSV for the enrichment pipeline to use.
                    sig_codes = [
                        (code, r) for code, r in code_enrich.items()
                        if r.get('p_value', 1.0) < 0.05
                    ]
                    sig_codes.sort(key=lambda x: x[1].get('q_value',
                                                           x[1]['p_value']))
                    if sig_codes and mappings:
                        go_path = os.path.join(
                            figures_dir, 'infection_genes_for_go.csv')
                        import csv
                        with open(go_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                ['vq_code', 'p_value', 'q_value', 'fold_change',
                                 'infected_mean', 'noninfected_mean',
                                 'gene_indices', 'gene_names'])
                            for code, r in sig_codes:
                                genes = set()
                                for key in (set(sub.keys) &
                                            set(mappings['vq_to_gene'].keys())):
                                    genes.update(
                                        mappings['vq_to_gene'][key].get(code, []))
                                gene_idx_list = sorted(genes)
                                gene_names_list = []
                                if mappings.get('gene_names'):
                                    for g in gene_idx_list:
                                        if g < len(mappings['gene_names']):
                                            gene_names_list.append(
                                                mappings['gene_names'][g])
                                writer.writerow([
                                    code, f"{r['p_value']:.6f}",
                                    f"{r.get('q_value', r['p_value']):.6f}",
                                    f"{r['fold_change']:.4f}",
                                    f"{r['infected_mean']:.6f}",
                                    f"{r['noninfected_mean']:.6f}",
                                    ';'.join(str(g) for g in gene_idx_list),
                                    ';'.join(gene_names_list),
                                ])
                        n_sig = len(sig_codes)
                        print(f"    Exported {n_sig} significant codes "
                              f"(p<0.05) to infection_genes_for_go.csv")

            # -- Sex code enrichment (sex_year_lake only) --
            if strat == 'sex_year_lake':
                sex_enrich = sub.sex_code_enrichment(
                    codebook_size=args.codebook_size)
                if sex_enrich:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    LakePlotter.sex_codes(sex_enrich, top_n=20, ax=ax)
                    fig.savefig(os.path.join(figures_dir,
                                'sex_code_enrichment.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print("    Saved sex_code_enrichment.png")

                    # Export gene lists for GO analysis (raw p<0.05 — see
                    # infection-code comment above for rationale).
                    sig_sex = [(code, r) for code, r in sex_enrich.items()
                               if r.get('p_value', 1.0) < 0.05]
                    sig_sex.sort(key=lambda x: x[1].get('q_value',
                                                        x[1]['p_value']))
                    if sig_sex and mappings:
                        go_path = os.path.join(
                            figures_dir, 'sex_genes_for_go.csv')
                        import csv as _csv
                        with open(go_path, 'w', newline='') as f:
                            writer = _csv.writer(f)
                            writer.writerow(
                                ['vq_code', 'p_value', 'q_value', 'fold_change',
                                 'male_mean', 'female_mean',
                                 'gene_indices', 'gene_names'])
                            for code, r in sig_sex:
                                genes = set()
                                for key in (set(sub.keys) &
                                            set(mappings['vq_to_gene'].keys())):
                                    genes.update(
                                        mappings['vq_to_gene'][key].get(code, []))
                                gene_idx_list = sorted(genes)
                                gene_names_list = []
                                if mappings.get('gene_names'):
                                    for g in gene_idx_list:
                                        if g < len(mappings['gene_names']):
                                            gene_names_list.append(
                                                mappings['gene_names'][g])
                                writer.writerow([
                                    code, f"{r['p_value']:.6f}",
                                    f"{r.get('q_value', r['p_value']):.6f}",
                                    f"{r['fold_change']:.4f}",
                                    f"{r['male_mean']:.6f}",
                                    f"{r['female_mean']:.6f}",
                                    ';'.join(str(g) for g in gene_idx_list),
                                    ';'.join(gene_names_list),
                                ])
                        print(f"    Exported {len(sig_sex)} significant codes "
                              f"(p<0.05) to sex_genes_for_go.csv")

            # -- Role (Source/Recipient) code enrichment (year_lake only) --
            if strat == 'year_lake':
                role_enrich = sub.role_code_enrichment(
                    codebook_size=args.codebook_size)
                if role_enrich:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    LakePlotter.role_codes(role_enrich, top_n=20, ax=ax)
                    fig.savefig(os.path.join(figures_dir,
                                'role_code_enrichment.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print("    Saved role_code_enrichment.png")

                    # Export gene lists for GO analysis (raw p<0.05 — see
                    # infection-code comment above for rationale).
                    sig_role = [(code, r) for code, r in role_enrich.items()
                                if r.get('p_value', 1.0) < 0.05]
                    sig_role.sort(key=lambda x: x[1].get('q_value',
                                                         x[1]['p_value']))
                    if sig_role and mappings:
                        go_path = os.path.join(
                            figures_dir, 'role_genes_for_go.csv')
                        import csv as _csv
                        with open(go_path, 'w', newline='') as f:
                            writer = _csv.writer(f)
                            writer.writerow(
                                ['vq_code', 'p_value', 'q_value', 'fold_change',
                                 'source_mean', 'recipient_mean',
                                 'gene_indices', 'gene_names'])
                            for code, r in sig_role:
                                genes = set()
                                for key in (set(sub.keys) &
                                            set(mappings['vq_to_gene'].keys())):
                                    genes.update(
                                        mappings['vq_to_gene'][key].get(code, []))
                                gene_idx_list = sorted(genes)
                                gene_names_list = []
                                if mappings.get('gene_names'):
                                    for g in gene_idx_list:
                                        if g < len(mappings['gene_names']):
                                            gene_names_list.append(
                                                mappings['gene_names'][g])
                                writer.writerow([
                                    code, f"{r['p_value']:.6f}",
                                    f"{r.get('q_value', r['p_value']):.6f}",
                                    f"{r['fold_change']:.4f}",
                                    f"{r['source_mean']:.6f}",
                                    f"{r['recipient_mean']:.6f}",
                                    ';'.join(str(g) for g in gene_idx_list),
                                    ';'.join(gene_names_list),
                                ])
                        print(f"    Exported {len(sig_role)} significant codes "
                              f"(p<0.05) to role_genes_for_go.csv")

    # ---- PERMANOVA: variance decomposition on VQ embeddings ----
    # Runs three decompositions per stratification:
    #   all lakes, source lakes only, recipient lakes only.
    # Factors include lake_category (Source/Recipient), ecotype, ancestry
    # (GenotypePool), match status, lake, year, sex, infection.
    print("\nPERMANOVA variance decomposition...")
    permanova_path = os.path.join(args.output_dir,
                                   _joint_path(args, 'permanova.pkl'))
    if args.force or not os.path.exists(permanova_path):
        import re as _re2
        permanova_results = {}

        def _build_metadata(sub, sd_obj):
            """Build per-key metadata dicts for PERMANOVA."""
            metadata = []
            for k in sub.keys:
                s = str(k)
                lake = s.split(' (')[0] if ' (' in s else s
                # Match both integer '(2023)' and float '(2023.0)' year
                # formats — keys are formatted via f'{lake} ({yr})' where yr
                # may be int or float depending on the metadata dtype. The
                # old '\.\d' regex required a decimal and silently returned
                # year=None (→ 'unknown') for integer years, inconsistent
                # with rol/analysis.py which parses via int(float(...)).
                year_match = _re2.search(r'\((\d{4}(?:\.\d+)?)\)', s)
                year = int(float(year_match.group(1))) if year_match else None
                meta = {
                    'Lake': lake,
                    'Year': str(year) if year is not None else 'unknown',
                    'Lake Category': str(sd_obj.get_lake_role(lake)),
                    # Ecotype (ancestry) and Lake Habitat are SEPARATE factors:
                    # a recipient lake's physical habitat can differ from the
                    # ancestry of its transplanted fish (e.g. Fred/Ranchero are
                    # Limnetic-ancestry but Benthic-habitat).  Source lakes fall
                    # back to habitat == ecotype.
                    'Ecotype': str(sd_obj.get_lake_ecotype(lake)),
                    'Lake Habitat': str(sd_obj.get_lake_habitat(lake)),
                }
                # Ancestry (GenotypePool) — meaningful for recipient lakes
                anc = sd_obj.get_genotype(lake)
                if anc and anc != 'nan':
                    meta['Ancestry'] = anc
                # Sex suffix
                if s.endswith('-f'):
                    meta['Sex'] = 'Female'
                elif s.endswith('-m'):
                    meta['Sex'] = 'Male'
                # Infection suffix
                inf_match = _re2.search(r'\)-([01])$', s)
                if inf_match:
                    meta['Infection'] = (
                        'Infected' if inf_match.group(1) == '1'
                        else 'Non-infected')
                metadata.append(meta)
            return metadata

        def _print_permanova(res, label):
            if not res:
                return
            print(f"  {label}:")
            for factor, d in res.items():
                q = d.get('q_value', d['p_value'])
                # Stars use the FDR-corrected q_value across the factor family.
                stars = ('***' if q < 0.001
                         else '**' if q < 0.01
                         else '*' if q < 0.05
                         else '')
                print(f"    {factor}: R²={d['r2']:.4f}, "
                      f"p={d['p_value']:.4f}, q={q:.4f} {stars}")

        for strat in strat_types:
            sub = analyzer.for_stratification(strat)
            if not sub.keys:
                continue

            # --- All lakes ---
            meta_all = _build_metadata(sub, sd)
            res_all = sub.permanova_decomposition(meta_all,
                                                   n_permutations=1000)
            if res_all:
                permanova_results[strat] = res_all
                _print_permanova(res_all, strat)

            # --- Source lakes only ---
            sub_src = sub.for_role('Source')
            if len(sub_src.keys) >= 3:
                meta_src = _build_metadata(sub_src, sd)
                res_src = sub_src.permanova_decomposition(
                    meta_src, n_permutations=1000)
                if res_src:
                    key = f'{strat} (Source only)'
                    permanova_results[key] = res_src
                    _print_permanova(res_src, key)

            # --- Recipient lakes only ---
            sub_rec = sub.for_role('Recipient')
            if len(sub_rec.keys) >= 3:
                meta_rec = _build_metadata(sub_rec, sd)
                res_rec = sub_rec.permanova_decomposition(
                    meta_rec, n_permutations=1000)
                if res_rec:
                    key = f'{strat} (Recipient only)'
                    permanova_results[key] = res_rec
                    _print_permanova(res_rec, key)

        save_pickle(permanova_results, permanova_path)
        print(f"  Saved permanova.pkl")
    else:
        permanova_results = load_pickle(permanova_path)
        print("  Loaded cached permanova.pkl")

    # PERMANOVA figure
    if permanova_results:
        fig = LakePlotter.permanova_bar(permanova_results)
        fig.savefig(os.path.join(figures_dir, 'permanova.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  Saved permanova.png")

    # ---- Thresholded correlation baseline ----
    print("\nThresholded correlation baseline...")
    threshold_path = os.path.join(args.output_dir,
                                   _joint_path(args, 'threshold_baseline.pkl'))
    if args.force or not os.path.exists(threshold_path):
        import re as _re3
        # Load a few expression matrices; matrices.pkl can be large, so
        # we read only what we need.  If unavailable, skip gracefully.
        _matrices_path = checkpoint_path(args.output_dir, 'matrices.pkl')
        if not os.path.exists(_matrices_path):
            print("  matrices.pkl not found — skipping threshold baseline")
            threshold_results = {}
        else:
            import pickle as _pickle
            with open(_matrices_path, 'rb') as _f:
                _all_matrices = _pickle.load(_f)
            sample_keys = sorted(_all_matrices.keys())[:5]
            threshold_results = {}
            for key in sample_keys:
                M = np.asarray(_all_matrices[key], dtype=np.float64)
                n_genes = M.shape[1]
                M_c = M - M.mean(axis=0, keepdims=True)
                M_std = M.std(axis=0, ddof=1)
                M_std[M_std == 0] = 1.0
                corr = (M_c.T @ M_c) / (M.shape[0] - 1)
                corr /= np.outer(M_std, M_std)
                upper_tri = corr[np.triu_indices(n_genes, k=1)]
                threshold = float(np.percentile(np.abs(upper_tri), 99))
                n_edges = int(np.sum(np.abs(corr) >= threshold)) // 2
                clean = _re3.sub(r'(\d{4})\.0', r'\1', str(key))
                threshold_results[clean] = {
                    'n_genes': n_genes,
                    'n_edges': n_edges,
                    'density': n_edges / (n_genes * (n_genes - 1) / 2),
                    'threshold_99': threshold,
                }
                print(f"  {clean}: {n_genes} genes, "
                      f"|r|≥{threshold:.4f}, {n_edges} edges "
                      f"({100*threshold_results[clean]['density']:.2f}%)")
            save_pickle(threshold_results, threshold_path)
            print(f"  Saved threshold_baseline.pkl")
    else:
        threshold_results = load_pickle(threshold_path)
        print("  Loaded cached threshold_baseline.pkl")

    # Threshold baseline figure
    if threshold_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        keys_list = list(threshold_results.keys())
        densities = [threshold_results[k]['density'] for k in keys_list]
        thresholds = [threshold_results[k]['threshold_99'] for k in keys_list]
        x = np.arange(len(keys_list))
        ax.bar(x, densities, color='#555555', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(keys_list, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Edge Density (top 1% |r|)')
        ax.set_title('Hard-Thresholded Correlation Baseline')
        for i, (d, t) in enumerate(zip(densities, thresholds)):
            ax.text(i, d + 0.0002, f'|r|≥{t:.3f}', ha='center', fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, 'threshold_baseline.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  Saved threshold_baseline.png")

    print(f"  All figures saved to {figures_dir}/")


# ---------------------------------------------------------------------------
# Step 10: Post-processing — paper-ready stats, tables, and plots
# ---------------------------------------------------------------------------

def step_postprocess(args):
    """Run post-processing scripts to produce paper-ready outputs.

    Executes (in order):
      - extract_stats.py  → stats.json (formatted statistics)
      - extract_enrichments.py → cluster enrichment tables
      - temporal_summary.py → temporal divergence jitter plots
      - eigenvalue_elbow.py → scree/elbow plot (reviewer response)
      - sample_size_table.py → LaTeX sample size table
      - extract_manuscript_stats.py → manuscript_stats.json (consolidated)
    """
    import subprocess

    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'scripts')
    out_dir = os.path.abspath(args.output_dir)

    post_dir = os.path.join(out_dir, 'postprocess')
    os.makedirs(post_dir, exist_ok=True)

    def _run(script_name, extra_args=None):
        script_path = os.path.join(scripts_dir, script_name)
        if not os.path.exists(script_path):
            print(f"  SKIP {script_name} — script not found at {script_path}")
            return
        cmd = [sys.executable, '-u', script_path,
               '--output-dir', out_dir]
        if extra_args:
            cmd.extend(extra_args)
        print(f"  Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, timeout=600)
        except subprocess.TimeoutExpired:
            print(f"  WARNING: {script_name} timed out after 10 min")
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: {script_name} failed (exit {e.returncode})")

    print("\n[Step 10] Post-processing — paper-ready outputs...")

    scripts_to_run = [
        ('extract_stats.py', 'stats.json'),
        ('extract_enrichments.py', 'cluster enrichments'),
        ('temporal_summary.py', 'temporal summary'),
        ('eigenvalue_elbow.py', 'elbow plot'),
        ('sample_size_table.py', 'sample size table'),
        ('extract_manuscript_stats.py', 'manuscript_stats.json'),
    ]

    for script_name, label in tqdm(scripts_to_run, desc="  Post-processing"):
        tqdm.write(f"  [{label}] {script_name}")
        if script_name == 'extract_stats.py':
            # JSON mode: capture stdout
            script_path = os.path.join(scripts_dir, script_name)
            if not os.path.exists(script_path):
                print(f"  SKIP {script_name} — not found")
                continue
            try:
                result = subprocess.run(
                    [sys.executable, '-u', script_path,
                     '--output-dir', out_dir, '--json'],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0 and result.stdout.strip():
                    json_path = os.path.join(post_dir, 'stats.json')
                    with open(json_path, 'w') as f:
                        f.write(result.stdout.strip())
                    print(f"  Saved stats.json ({len(result.stdout)} bytes)")
                else:
                    print(f"  WARNING: extract_stats.py exit {result.returncode}")
            except Exception as e:
                print(f"  WARNING: extract_stats.py failed: {e}")
        elif script_name == 'eigenvalue_elbow.py':
            matrices_path = os.path.join(out_dir, 'matrices.pkl')
            _run(script_name, extra_args=['--matrices', matrices_path])
        else:
            _run(script_name)

    # Move any generated outputs into postprocess/
    for fn in os.listdir(out_dir):
        src = os.path.join(out_dir, fn)
        if os.path.isfile(src) and fn not in os.listdir(post_dir):
            if any(fn.endswith(ext) for ext in
                   ['.json', '_enrichments.csv', '_summary.png',
                    'elbow_', 'sample_size_', 'temporal_']):
                import shutil
                dst = os.path.join(post_dir, fn)
                shutil.move(src, dst)
                print(f"  Moved {fn} → postprocess/")

    print(f"  Post-processing complete. Outputs in {post_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='RoL: Full pipeline with checkpointing'
    )

    # Data paths
    parser.add_argument('--transcriptome', default='data/7.HKTranscriptome.csv',
                        help='Path to transcriptome CSV')
    parser.add_argument('--metadata', default='data/rawCount_metadata_2019_2023.csv',
                        help='Path to metadata CSV')
    parser.add_argument('--morphology', default=None,
                        help='Path to morphology CSV (optional)')
    parser.add_argument('--infection', default=None,
                        help='Path to infection CSV (optional)')
    parser.add_argument('--output-dir', default='output',
                        help='Directory for checkpoints and results')

    # Batch correction (optional Step 0)
    parser.add_argument('--batch-correct', action='store_true',
                        help='Run VST + ComBat batch correction before pipeline')
    parser.add_argument('--batch-transcriptome',
                        default='data/7.HKTranscriptome.csv',
                        help='Path to raw HK transcriptome for batch correction')
    parser.add_argument('--batch-metadata', default=None,
                        help='Metadata path for batch correction '
                             '(default: same as --metadata)')
    parser.add_argument('--batch-morphology', default=None,
                        help='Morphology path for batch correction '
                             '(default: same as --morphology)')
    parser.add_argument('--batch-correct-output', default=None,
                        help='Output dir for batch correction '
                             '(default: <output-dir>/results_batch_corrected)')
    parser.add_argument('--batch-n-genes', type=int, default=10000,
                        help='Number of top genes retained during batch '
                             'correction (0 = all, default 10000)')

    # Model hyperparameters
    parser.add_argument('--device', default='cpu',
                        help='Device for training/inference (cpu/cuda)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs per graph')
    parser.add_argument('--resume', action='store_true',
                        help='Resume batched training from existing model_last.pt '
                             '/ *_opt.pt checkpoints instead of restarting. '
                             '--epochs is the TOTAL epoch count: training '
                             'continues from the last completed epoch up to '
                             '--epochs. Only applies to --train-both + '
                             '--graph-batch-size (batched) mode.')
    parser.add_argument('--start-epoch', type=int, default=None,
                        help='Override the number of already-completed epochs '
                             'when resuming (training resumes at the next one). '
                             'Use for a legacy checkpoint that has no '
                             'train_meta.pkl and the best_epoch fallback is wrong.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--commit-alpha', type=float, default=0.25,
                        help='VQ commitment loss weight')
    parser.add_argument('--noise-scale', type=float, default=5.0,
                        help='Multiplier for radii (0 disables noise)')
    parser.add_argument('--codebook-size', type=int, default=100,
                        help='Number of VQ codes in the codebook')

    # Gene filtering (two-stage: expression → variance)
    parser.add_argument('--min-expression', type=float, default=0.0,
                        help='Remove genes with mean expression below this '
                             'threshold before variance filtering. '
                             'Default: 0 (no expression filter). '
                             'Set to ~1e-5 to remove very lowly-expressed genes.')
    parser.add_argument('--top-n-genes', type=int, default=None,
                        help='After expression filtering, keep the top N genes '
                             'by variance across all samples. '
                             'Default: use all surviving genes.')

    # Analysis parameters
    parser.add_argument('--base-year', type=int, default=2019,
                        help='Reference year for temporal drift')
    parser.add_argument('--linkage-method', default='ward',
                        choices=['ward', 'average', 'complete', 'single'],
                        help='Linkage method for hierarchical clustering')
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='Number of clusters to cut dendrogram (None = no cut)')

    # Permutation parameters
    parser.add_argument('--label-permutations', type=int, default=1000,
                        help='Permutations for label-shuffling test')

    # Training mode
    parser.add_argument('--train-joint', action='store_true',
                        help='Use train_joint() with lake-identity conditioning '
                             'instead of sequential per-graph training.'
                             'Model saved to model_joint.pt. '
                             'Sample permutations (step 9) are skipped.')
    parser.add_argument('--train-both', action='store_true',
                        help='Train BOTH sequential and joint paradigms in one '
                             'run. Graphs loaded once, then sequential trained '
                             'first, then joint from the same in-memory data. '
                             'Saves both model.pt and model_joint.pt.')
    parser.add_argument('--graph-batch-size', type=int, default=0,
                        help='Load graphs in batches of this size instead of all '
                             'at once. 0 = load all (default). Set to e.g. 50 '
                             'when training with many genes (>10k) to limit CPU '
                             'RAM. Only applies to --train-both mode.')
    # Pipeline control
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--force', action='store_true',
                        help='Recompute all steps (ignore checkpoints)')
    parser.add_argument('--graphs-dir', type=str, default=None,
                        help='Path to directory with pre-built graphs, matrices, '
                             'and data pickles. When set, skips Steps 0-2 and '
                             'loads from this directory instead.')
    parser.add_argument('--stop-after', type=int, default=None,
                        choices=[0, 2, 3, 5, 7, 8, 9, 10],
                        help='Stop pipeline after given step number '
                             '(0=batch, 2=graphs, 3=training, 5=mappings, '
                             '7=clustering, 8=label-perm, 9=figures, 10=all)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration derived from data — n_genes is set automatically
    cfg = {
        'stratifications': ['year_lake', 'sex_year_lake', 'infection_year_lake'],
        'n_eigencomponents': 17,
        'reconstruction_levels': [2, 4, 16],
        'in_channels': 64,
        'hidden_channels': 64,
        'out_channels': 8,
        'num_layers': 3,
        'dropout': 0.2,
        'codebook_channels': 8,
        'codebook_size': args.codebook_size,
        'decoder_channels': 64,
    }

    t0 = time.time()

    if args.graphs_dir:
        # --- Skip Steps 0-2: load pre-built artefacts ---
        print(f"[Skip 0-2] Loading pre-built graphs from {args.graphs_dir}")
        matrices = load_pickle(
            os.path.join(args.graphs_dir, 'matrices.pkl'))
        # Pop metadata key if present
        matrices.pop('__meta__', None)
        graphs = load_pickle(
            os.path.join(args.graphs_dir, 'graphs_manifest.pkl'))
        radii = load_pickle(
            os.path.join(args.graphs_dir, 'radii.pkl'))

        # Rewrite graph paths (stored as absolute Modal paths) to be
        # relative to args.graphs_dir
        graphs_subdir = os.path.join(args.graphs_dir, 'graphs')
        for key in list(graphs.keys()):
            val = graphs[key]
            if isinstance(val, str) and '/vol/output/graphs/' in val:
                fname = os.path.basename(val)
                graphs[key] = os.path.join(graphs_subdir, fname)

        # Infer n_genes from any expression matrix
        sample_mat = next(iter(matrices.values()))
        if hasattr(sample_mat, 'shape'):
            n_genes = sample_mat.shape[1]
        else:
            n_genes = len(sample_mat[0]) if len(sample_mat) > 0 else 0
        cfg['n_genes'] = n_genes

        # Load SticklebackData (may fail with pandas version mismatch;
        # downstream steps that need sd will error with a clear message)
        data_path = os.path.join(args.graphs_dir, 'data.pkl')
        sd = None
        if os.path.exists(data_path):
            try:
                sd = load_pickle(data_path)
            except Exception as e:
                print(f"  WARNING: could not load data.pkl ({e})")
                print(f"  Downstream steps that require metadata will fail.")
        print(f"  Loaded {len(matrices)} matrices, "
              f"{len(graphs)} graphs, {n_genes} genes")
        # _validate_graphs deferred — loads every .pt file via torch.load
        # _validate_graphs(graphs, label="pre-built graphs")
    else:
        # --- Step 0: Batch correction (optional) ---
        if args.batch_correct:
            if not os.path.exists(args.batch_transcriptome):
                print(f"[Step 0] ERROR: Batch transcriptome not found: "
                      f"{args.batch_transcriptome}")
                sys.exit(1)
            args.transcriptome = step_batch_correct(args)

        # --- Steps 1-2: Data → Graphs ---
        matrices, sd = step_load_data(args, cfg)
        cfg['n_genes'] = sd.n_genes
        graphs, radii = step_build_graphs(args, cfg, matrices)

    # --- Optional: two-stage gene filtering ---
    # Stage 1: remove lowly-expressed genes (technical noise)
    # Stage 2: keep top N by variance (biologically informative fraction)
    do_expression_filter = args.min_expression > 0
    do_variance_filter = (
        args.top_n_genes is not None and args.top_n_genes < cfg['n_genes']
    )

    if do_expression_filter or do_variance_filter:
        # Pool expression data across all matrices for gene-level statistics
        all_expr = []
        for M in matrices.values():
            if hasattr(M, 'shape') and M.shape[0] > 0:
                all_expr.append(np.asarray(M, dtype=np.float64))
        pooled = np.vstack(all_expr) if all_expr else None

        if pooled is None:
            print("  WARNING: No expression data to filter — skipping")
        else:
            n_original = pooled.shape[1]
            gene_means = np.mean(pooled, axis=0)
            gene_vars = np.var(pooled, axis=0)
            keep_mask = np.ones(n_original, dtype=bool)

            # --- Stage 1: expression filter ---
            if do_expression_filter:
                expr_keep = gene_means >= args.min_expression
                n_removed = int((~expr_keep).sum())
                keep_mask &= expr_keep
                print(f"\n  Stage 1 (expression filter): removed {n_removed} "
                      f"genes with mean < {args.min_expression:.2e} "
                      f"({100*n_removed/n_original:.1f}%)")
                print(f"    Surviving: {int(keep_mask.sum())} genes "
                      f"(mean range: {gene_means[keep_mask].min():.2e} – "
                      f"{gene_means[keep_mask].max():.2e})")

            # --- Stage 2: variance filter ---
            if do_variance_filter:
                surviving_idx = np.where(keep_mask)[0]
                n_surviving = len(surviving_idx)
                if args.top_n_genes >= n_surviving:
                    print(f"  Stage 2 (variance filter): --top-n-genes "
                          f"({args.top_n_genes}) >= surviving genes "
                          f"({n_surviving}) — keeping all")
                else:
                    # Rank surviving genes by variance (descending)
                    surv_vars = gene_vars[surviving_idx]
                    var_order = np.argsort(surv_vars)[::-1]
                    top_surviving = surviving_idx[var_order[:args.top_n_genes]]
                    new_mask = np.zeros(n_original, dtype=bool)
                    new_mask[top_surviving] = True
                    keep_mask = new_mask
                    print(f"  Stage 2 (variance filter): kept top "
                          f"{args.top_n_genes} genes by variance "
                          f"(variance range: "
                          f"{gene_vars[keep_mask].min():.2e} – "
                          f"{gene_vars[keep_mask].max():.2e})")

            # --- Apply mask to all matrices ---
            final_idx = np.where(keep_mask)[0]
            final_idx = np.sort(final_idx)
            print(f"  Final gene set: {len(final_idx)}/{n_original} "
                  f"({100*len(final_idx)/n_original:.1f}%)")

            for key in tqdm(list(matrices.keys()), desc="  Subsetting matrices"):
                M = np.asarray(matrices[key])
                matrices[key] = M[:, final_idx]

            if sd is not None:
                sd.gene_names = [sd.gene_names[i] for i in final_idx]
                sd.n_genes = len(final_idx)
            cfg['n_genes'] = len(final_idx)

            if args.graphs_dir:
                print("  WARNING: --graphs-dir graphs were built with full "
                      "gene set. Gene filtering with pre-built graphs "
                      "will cause dimension mismatch. "
                      "Rebuild graphs without --graphs-dir.")

    if args.stop_after is not None and args.stop_after <= 2:
        elapsed = time.time() - t0
        print(f"\nPipeline stopped after step 2 in {elapsed:.0f}s. "
              f"Results in {args.output_dir}/")
        return

    # --- Step 3-9: Training → Figures ---
    # When --train-both, pre-load graphs once then run both paradigms.
    if args.train_both:
        print("\n" + "=" * 60)
        print("  --train-both: loading graphs once, training both paradigms")
        print("=" * 60)
        lake_names = sorted(set(
            str(k).split(' (')[0] for k in graphs.keys()
        ))
        lake_name_to_id = {name: i for i, name in enumerate(lake_names)}
        n_lakes = len(lake_names)
        all_keys = sorted(graphs.keys())

        if args.graph_batch_size > 0:
            # --- Batched mode: load subsets to limit CPU memory ---
            print(f"  Using batched loading: {args.graph_batch_size} graphs/batch")
            seq_model, joint_model = _train_models_batched(
                args, cfg, graphs, radii, lake_name_to_id, n_lakes,
                all_keys, args.graph_batch_size)

            paradigms = [
                (seq_model, False, 'Sequential'),
                (joint_model, True, 'Joint'),
            ]
            for model, _, label in paradigms:
                print(f"\n{'=' * 60}")
                print(f"  {label} downstream (steps 4-9)")
                print(f"{'=' * 60}")
                args.train_joint = (label == 'Joint')
                embeddings = step_generate_embeddings(args, graphs, model)
                mappings = step_build_gene_mappings(args, graphs, model, sd)
                slopes = step_temporal_slopes(args, embeddings, sd)
                clustering = step_hierarchical_clustering(args, embeddings, sd)
                label_perm = step_label_permutation(args, embeddings, sd)
                step_generate_figures(
                    args, embeddings, sd, mappings, slopes, clustering,
                    label_perm, model=model,
                )
        else:
            # --- Original: pre-load all graphs, then train sequentially ---
            preloaded = _preload_all_graphs(
                graphs, radii, args.noise_scale,
                lake_name_to_id=lake_name_to_id)
            print(f"  Pre-loaded {preloaded['n_loaded']} graphs "
                  f"({preloaded['n_loaded']} valid, "
                  f"{len(graphs) - preloaded['n_loaded']} degenerate)")

            paradigms = [
                (False, 'Sequential', 'model.pt'),
                (True, 'Joint', 'model_joint.pt'),
            ]

            for train_joint_flag, label, _ in paradigms:
                print(f"\n{'=' * 60}")
                print(f"  {label} training")
                print(f"{'=' * 60}")
                args.train_joint = train_joint_flag

                model = step_train_model(args, cfg, graphs, radii,
                                         preloaded=preloaded)
                if model is None:
                    continue

                embeddings = step_generate_embeddings(args, graphs, model)
                mappings = step_build_gene_mappings(args, graphs, model, sd)
                slopes = step_temporal_slopes(args, embeddings, sd)
                clustering = step_hierarchical_clustering(args, embeddings, sd)
                label_perm = step_label_permutation(args, embeddings, sd)
                step_generate_figures(
                    args, embeddings, sd, mappings, slopes, clustering,
                    label_perm, model=model,
                )

    else:
        # --- Step 3: Model training (single paradigm) ---
        model = step_train_model(args, cfg, graphs, radii)

        if args.stop_after is not None and args.stop_after <= 3:
            elapsed = time.time() - t0
            print(f"\nPipeline stopped after step 3 in {elapsed:.0f}s. "
                  f"Results in {args.output_dir}/")
            return

        # --- Steps 4-5: Embeddings → Mappings ---
        embeddings = step_generate_embeddings(args, graphs, model)
        mappings = step_build_gene_mappings(args, graphs, model, sd)

        if args.stop_after is not None and args.stop_after <= 5:
            elapsed = time.time() - t0
            print(f"\nPipeline stopped after step 5 in {elapsed:.0f}s. "
                  f"Results in {args.output_dir}/")
            return

        # --- Steps 6-7: Analysis ---
        slopes = step_temporal_slopes(args, embeddings, sd)
        clustering = step_hierarchical_clustering(args, embeddings, sd)

        if args.stop_after is not None and args.stop_after <= 7:
            elapsed = time.time() - t0
            print(f"\nPipeline stopped after step 7 in {elapsed:.0f}s. "
                  f"Results in {args.output_dir}/")
            return

        # --- Step 8: Label-permutation test ---
        label_perm = step_label_permutation(args, embeddings, sd)

        if args.stop_after is not None and args.stop_after <= 8:
            elapsed = time.time() - t0
            print(f"\nPipeline stopped after step 8 in {elapsed:.0f}s. "
                  f"Results in {args.output_dir}/")
            return

        # --- Step 9: Generate figures ---
        step_generate_figures(
            args, embeddings, sd, mappings, slopes, clustering,
            label_perm, model=model,
        )

        if args.stop_after is not None and args.stop_after <= 9:
            elapsed = time.time() - t0
            print(f"\nPipeline stopped after step 9 in {elapsed:.0f}s. "
                  f"Results in {args.output_dir}/")
            return

    # --- Step 10: Post-processing (paper-ready stats + tables) ---
    step_postprocess(args)

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.0f}s. Results in {args.output_dir}/")


if __name__ == '__main__':
    main()
