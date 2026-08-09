"""
Modal deployment for RoL pipeline.

Two main entrypoints for end-to-end paper-ready outputs:

  1. Joint (lake-conditioned) training:
       modal run modal_pipeline.py::joint
       modal run modal_pipeline.py::joint --top-n-genes 10000

  2. Sequential (per-graph) training:
       modal run modal_pipeline.py::sequential
       modal run modal_pipeline.py::sequential --top-n-genes 10000

Cleanup before retraining:
       modal run modal_pipeline.py::cleanup

Other utilities:
       modal run modal_pipeline.py::wgcna_full
       modal run modal_pipeline.py::run_original_mappings

Shared Modal volume ``rol_output`` bridges checkpoints between stages.
Upload data once via CLI:

  modal volume put rol_output data/1.Metadata.csv data/7.HKTranscriptome.csv \
      data/2.Morphology.csv data/3.Infection.csv \
      output/gene_name_to_locid.tsv

Pricing (approx):
  CPU-only (64 GB, 8 cores):  ~$0.50 / h
  A100 80 GB:                 ~$2.10 / h
"""

import modal
import os
import json
import time
import uuid

# ---------------------------------------------------------------------------
# Shared volume — holds input data AND output checkpoints
# ---------------------------------------------------------------------------
# Create once:  modal volume create rol_output
volume = modal.Volume.from_name("rol_output", create_if_missing=True)

# Paths inside the volume
DATA_DIR   = "/vol"        # files at volume root, not in a subdir
OUTPUT_DIR = "/vol/25k_v5"

# ---------------------------------------------------------------------------
# Concurrency guard: all entrypoints share OUTPUT_DIR, so two simultaneous
# `modal run` invocations (e.g. joint + sequential) would silently clobber
# each other's checkpoints. A sentinel file coordinates them: a run holds the
# lock for the duration of its cpu_steps + train_steps phases (sharing one
# run_id), and any other entrypoint aborts loudly until the lock is released.
# ---------------------------------------------------------------------------
_RUN_LOCK = ".active_run"
_STALE_SECS = 25 * 3600      # just over the 24h Modal step timeout


def _acquire_run_lock(run_id, stale_secs=_STALE_SECS):
    """Claim OUTPUT_DIR for ``run_id``; abort if a different fresh run holds it.

    Call ``volume.reload()`` before this so the latest committed sentinel is
    visible. The cpu_steps and train_steps phases of ONE entrypoint pass the
    same run_id, so the normal serial flow never blocks itself.

    ATOMICITY NOTE: Modal volume writes are local to each container until
    ``volume.commit()``, so a pure check-then-write is NOT a cross-container
    mutex — two containers can both pass the check and both commit.  This
    function mitigates with (1) an atomic O_EXCL local create and (2) a
    post-commit revalidation that aborts a run whose lock was overwritten by a
    concurrent entrypoint.  This cannot be a perfect mutex on a shared volume;
    the window is small but real.  Do NOT rely on it for truly concurrent runs.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lock_path = os.path.join(OUTPUT_DIR, _RUN_LOCK)

    def _read_lock():
        try:
            with open(lock_path) as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    info = _read_lock()
    other = info.get('run_id')
    age = time.time() - info.get('started_at', 0)
    if other and other != run_id:
        if age < stale_secs:
            raise RuntimeError(
                f"Another RoL run (run_id={other}) is active in {OUTPUT_DIR} "
                f"(started {age/3600:.1f}h ago). Concurrent entrypoints share "
                f"this output dir and would silently corrupt each other's "
                f"checkpoints. Wait for it to finish; or, if it crashed, "
                f"remove {lock_path} and retry."
            )
        print(f"[lock] Stale run lock (run_id={other}, "
              f"{age/3600:.1f}h old) — overwriting")

    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        # A local/committed lock appeared between our read and create.
        raise RuntimeError(
            f"[lock] Lost the lock for {OUTPUT_DIR} to a concurrent "
            f"entrypoint during acquisition — aborting rather than clobbering.")
    with os.fdopen(fd, 'w') as fh:
        json.dump({'run_id': run_id, 'started_at': time.time()}, fh)
    volume.commit()

    # Post-commit revalidation: another container may have committed its own
    # lock after our create.  If so, we are the loser — abort loudly.
    volume.reload()
    info2 = _read_lock()
    if info2.get('run_id') != run_id:
        raise RuntimeError(
            f"[lock] Lost the lock to run_id={info2.get('run_id')} after "
            f"commit — a concurrent entrypoint overwrote it. Aborting.")


def _release_run_lock(run_id):
    """Release the lock if it still belongs to ``run_id``."""
    lock_path = os.path.join(OUTPUT_DIR, _RUN_LOCK)
    if not os.path.exists(lock_path):
        return
    try:
        with open(lock_path) as fh:
            info = json.load(fh)
    except Exception:
        info = {}
    if info.get('run_id') == run_id:
        os.remove(lock_path)
        volume.commit()


def _check_no_active_run():
    """Abort if a fresh run is in progress (used by cleanup)."""
    lock_path = os.path.join(OUTPUT_DIR, _RUN_LOCK)
    if not os.path.exists(lock_path):
        return
    try:
        with open(lock_path) as fh:
            info = json.load(fh)
    except Exception:
        info = {}
    age = time.time() - info.get('started_at', 0)
    if age < _STALE_SECS:
        raise RuntimeError(
            f"Refusing to clean: a RoL run (run_id={info.get('run_id')}) "
            f"appears active ({age/3600:.1f}h ago). cleanup would destroy an "
            f"in-flight run. If it crashed, remove {lock_path} first."
        )
    print(f"[lock] Stale run lock ({age/3600:.1f}h old) — proceeding")


# ---------------------------------------------------------------------------
# Enrichment summary writer — reads whatever WGCNA data was generated and
# writes readable markdown + JSON tables into the postprocess folder.
# ---------------------------------------------------------------------------

def _write_enrichment_summary(wgcna_dir, postprocess_dir):
    """Read enrichment CSVs from ``wgcna_dir`` and write summary tables."""
    import csv
    import json
    import os as _os

    _os.makedirs(postprocess_dir, exist_ok=True)

    summary_path = _os.path.join(wgcna_dir, 'wgcna_summary.json')
    if not _os.path.exists(summary_path):
        print("[wgcna_full] No wgcna_summary.json — skipping enrichment summary")
        return

    with open(summary_path) as f:
        wgcna_summary = json.load(f)

    md_lines = []
    json_comps = {}

    COMPARISONS = ['infection', 'sex', 'year']

    for comp in COMPARISONS:
        info = wgcna_summary.get(comp)
        if info is None:
            md_lines.append(f"## {comp.title()}\n\n*WGCNA failed — no data produced.*\n")
            json_comps[comp] = {'status': 'failed'}
            continue

        md_lines.append(f"## {info['label_a'].title()} vs {info['label_b'].title()}")
        md_lines.append(f"- {info['n_a']} fish ({info['label_a']}), "
                        f"{info['n_b']} fish ({info['label_b']})")
        md_lines.append(f"- Modules: {info['label_a']} {info['modules_a']}, "
                        f"{info['label_b']} {info['modules_b']}")
        md_lines.append(f"- Preserved: {info['n_preserved']}")
        md_lines.append(f"- Enriched terms (WGCNA): "
                        f"{info['n_enriched_terms_a']} ({info['label_a']}), "
                        f"{info['n_enriched_terms_b']} ({info['label_b']})")
        md_lines.append(f"- Enriched terms (VQ): {info['n_vq_enriched_terms']}")
        md_lines.append(f"- Shared GO terms: {info['n_shared_terms']}")
        md_lines.append("")

        json_comp = {
            'status': 'ok',
            'label_a': info['label_a'], 'label_b': info['label_b'],
            'n_fish_a': info['n_a'], 'n_fish_b': info['n_b'],
            'modules_a': info['modules_a'], 'modules_b': info['modules_b'],
            'n_preserved': info['n_preserved'],
            'eigengene_enrichments': {},
            'vq_enrichment': [],
            'term_overlap': {'shared': [], 'vq_only': [], 'wgcna_only': []},
        }

        comp_dir = _os.path.join(wgcna_dir, comp)

        # --- eigengene enrichments (per-group WGCNA GO) ---
        enrich_dir = _os.path.join(comp_dir, 'eigengene_enrichments')
        for label in (info['label_a'], info['label_b']):
            csv_path = _os.path.join(enrich_dir, label,
                                     'eigengene_enrichments.csv')
            if not _os.path.exists(csv_path):
                continue
            rows = []
            with open(csv_path, newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            if rows:
                json_comp['eigengene_enrichments'][label] = rows
                md_lines.append(f"### WGCNA eigengene GO — {label}")
                md_lines.append("| Source | Term | p-value |")
                md_lines.append("|--------|------|---------|")
                for r in sorted(rows, key=lambda x: float(x.get('p_value', 1)))[:15]:
                    src = r.get('source', '')
                    tname = r.get('term_name', '')[:80]
                    pv = float(r.get('p_value', 1))
                    md_lines.append(f"| {src} | {tname} | {pv:.2e} |")
                md_lines.append("")

        # --- Gene-level overlap (VQ codes ↔ WGCNA modules) ---
        # Scan for paradigm-suffixed files (vq_wgcna_overlap_{comp}.csv for
        # backward compat, vq_wgcna_overlap_{comp}_sequential.csv etc. for
        # multi-paradigm runs).
        overlap_dir = _os.path.join(comp_dir, 'vq_wgcna_comparison')
        all_gene_overlap_rows = []
        if _os.path.isdir(overlap_dir):
            for fn in sorted(_os.listdir(overlap_dir)):
                if not (fn.startswith(f'vq_wgcna_overlap_{comp}')
                        and fn.endswith('.csv')):
                    continue
                # Derive paradigm from filename
                # vq_wgcna_overlap_infection.csv          → primary
                # vq_wgcna_overlap_infection_sequential.csv → sequential
                # vq_wgcna_overlap_infection_joint.csv      → joint
                stem = fn[:-4]  # strip .csv
                suffix = stem[len(f'vq_wgcna_overlap_{comp}'):]
                paradigm = suffix.lstrip('_') if suffix else 'primary'
                gene_ov_csv = _os.path.join(overlap_dir, fn)
                if not _os.path.exists(gene_ov_csv):
                    continue
                rows = []
                with open(gene_ov_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        r['_paradigm'] = paradigm
                        rows.append(r)
                if rows:
                    all_gene_overlap_rows.extend(rows)
        if all_gene_overlap_rows:
            json_comp['vq_wgcna_gene_overlap'] = all_gene_overlap_rows
            n_sig = sum(1 for r in all_gene_overlap_rows
                        if float(r.get('fisher_p', 1)) < 0.05)
            md_lines.append(f"### Gene overlap — VQ codes ↔ WGCNA modules "
                            f"({len(all_gene_overlap_rows)} pairs, {n_sig} significant)")
            md_lines.append("| VQ Code | Module | Paradigm | Overlap | Jaccard | Fisher p |")
            md_lines.append("|---------|--------|----------|---------|---------|----------|")
            for r in sorted(all_gene_overlap_rows, key=lambda x: -float(x.get('jaccard', 0)))[:20]:
                code = r.get('vq_code', '')
                mod = r.get('wgcna_module', '')[:40]
                par = r.get('_paradigm', '')[:1].upper() if r.get('_paradigm') else ''
                n_ov = r.get('n_overlap', '')
                j = float(r.get('jaccard', 0))
                fp = float(r.get('fisher_p', 1))
                md_lines.append(f"| {code} | {mod} | {par} | {n_ov} | {j:.3f} | {fp:.2e} |")
            md_lines.append("")

        # --- VQ code GO enrichment (step 8 — requires *_genes_for_go.csv) ---
        # Scan for vq_code_enrichment*.csv — may have paradigm suffix
        # (e.g. vq_code_enrichment_sequential.csv, vq_code_enrichment.csv)
        import re as _re
        all_vq_rows = []
        cand_files = []
        vq_term_dir = _os.path.join(comp_dir, 'vq_wgcna_term_comparison')
        for fn in (_os.listdir(vq_term_dir) if _os.path.isdir(vq_term_dir) else []):
            if fn.startswith('vq_code_enrichment') and fn.endswith('.csv'):
                cand_files.append(_os.path.join(vq_term_dir, fn))
        for vq_csv in cand_files:
            # Derive paradigm from filename: 'vq_code_enrichment.csv' → primary
            # (no suffix), 'vq_code_enrichment_sequential.csv' → sequential
            basename = _os.path.basename(vq_csv)
            m = _re.match(r'vq_code_enrichment(?:_(.+))?\.csv$', basename)
            paradigm = m.group(1) if m and m.group(1) else 'primary'
            if not _os.path.exists(vq_csv):
                continue
            rows = []
            with open(vq_csv, newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r['_paradigm'] = paradigm
                    rows.append(r)
            if rows:
                all_vq_rows.extend(rows)
                suffix = ' (S)' if paradigm == 'sequential' else ' (J)' if paradigm == 'joint' else ''
                md_lines.append(f"### VQ code GO enrichment — {paradigm}")
                md_lines.append("| Code | Source | Term | p-value |")
                md_lines.append("|------|--------|------|---------|")
                for r in sorted(rows, key=lambda x: float(x.get('p_value', 1)))[:15]:
                    code = r.get('vq_code', '')
                    src = r.get('source', '')
                    tname = r.get('term_name', '')[:80]
                    pv = float(r.get('p_value', 1))
                    code_display = f"{code}{suffix}"
                    md_lines.append(f"| {code_display} | {src} | {tname} | {pv:.2e} |")
                md_lines.append("")
        if all_vq_rows:
            json_comp['vq_enrichment'] = all_vq_rows

        # --- GO term overlap (shared / vq_only / wgcna_only) ---
        # Scan for paradigm-suffixed files so both paradigms' data are included.
        vq_term_dir = _os.path.join(comp_dir, 'vq_wgcna_term_comparison')
        all_term_overlap_rows = []
        if _os.path.isdir(vq_term_dir):
            for fn in sorted(_os.listdir(vq_term_dir)):
                if not (fn.startswith(f'vq_wgcna_term_overlap_{comp}')
                        and fn.endswith('.csv')):
                    continue
                # Derive paradigm from filename
                stem = fn[:-4]
                suffix = stem[len(f'vq_wgcna_term_overlap_{comp}'):]
                paradigm = suffix.lstrip('_') if suffix else 'primary'
                ov_csv = _os.path.join(vq_term_dir, fn)
                if not _os.path.exists(ov_csv):
                    continue
                with open(ov_csv, newline='') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        r['_paradigm'] = paradigm
                        all_term_overlap_rows.append(r)
        if all_term_overlap_rows:
            for cat, label in [('shared', 'Shared'),
                               ('vq_only', 'VQ-only'),
                               ('wgcna_only', 'WGCNA-only')]:
                subset = [r for r in all_term_overlap_rows if r.get('overlap') == cat]
                if subset:
                    json_comp['term_overlap'][cat] = subset
                    md_lines.append(f"### GO terms — {label} ({len(subset)})")
                    md_lines.append("| Source | Term | Paradigm | p-value |")
                    md_lines.append("|--------|------|----------|---------|")
                    pkey = ('vq_best_p' if cat == 'vq_only'
                            else ('wgcna_best_p' if cat == 'wgcna_only'
                                  else 'vq_best_p'))
                    for r in sorted(subset, key=lambda x: float(x.get(pkey, 1)))[:10]:
                        src = r.get('source', '')
                        tname = r.get('term_name', '')[:80]
                        pv = float(r.get(pkey, 1))
                        par = r.get('_paradigm', '')[:1].upper() if r.get('_paradigm') else ''
                        md_lines.append(f"| {src} | {tname} | {par} | {pv:.2e} |")
                    md_lines.append("")

        json_comps[comp] = json_comp

    # --- Role (Source/Recipient) VQ enrichment — VQ-only, no WGCNA ---
    # Scan for role* subdirs (role, role_sequential, role_joint)
    for role_subdir in sorted(_os.listdir(wgcna_dir) if _os.path.isdir(wgcna_dir) else []):
        if not role_subdir.startswith('role'):
            continue
        role_subdir_path = _os.path.join(wgcna_dir, role_subdir)
        if not _os.path.isdir(role_subdir_path):
            continue
        # Derive paradigm: 'role' → primary, 'role_sequential' → sequential, etc.
        paradigm = role_subdir.replace('role_', '') if role_subdir != 'role' else 'primary'
        role_csv = _os.path.join(role_subdir_path, 'vq_code_enrichment.csv')
        if not _os.path.exists(role_csv):
            continue
        rows = []
        with open(role_csv, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                r['_paradigm'] = paradigm
                rows.append(r)
        if rows:
            key = 'role' if paradigm == 'primary' else f'role_{paradigm}'
            json_comps[key] = {
                'status': 'ok',
                'label': 'Source vs Recipient',
                'paradigm': paradigm,
                'description': 'VQ-only g:Profiler enrichment (no WGCNA)',
                'vq_enrichment': rows,
            }
            p_suffix = ' (S)' if paradigm == 'sequential' else ' (J)' if paradigm == 'joint' else ''
            n_terms = len(rows)
            n_codes = len(set(r.get('vq_code', '') for r in rows))
            md_lines.append(f"## Source vs Recipient — VQ only ({paradigm})")
            md_lines.append(f"- {n_codes} significant codes, {n_terms} enriched terms")
            md_lines.append("| Code | Source | Term | p-value |")
            md_lines.append("|------|--------|------|---------|")
            for r in sorted(rows, key=lambda x: float(x.get('p_value', 1)))[:15]:
                code = r.get('vq_code', '')
                src = r.get('source', '')
                tname = r.get('term_name', '')[:80]
                pv = float(r.get('p_value', 1))
                code_display = f"{code}{p_suffix}"
                md_lines.append(f"| {code_display} | {src} | {tname} | {pv:.2e} |")
            md_lines.append("")
    has_role = any(k.startswith('role') for k in json_comps)
    if not has_role:
        json_comps['role'] = {'status': 'not_found'}

    # --- Write outputs ---
    md_path = _os.path.join(postprocess_dir, 'enrichment_summary.md')
    with open(md_path, 'w') as f:
        f.write('# WGCNA Enrichment Summary\n\n')
        for line in md_lines:
            f.write(line + '\n')
    print(f"[wgcna_full] Wrote enrichment summary to {md_path}")

    json_path = _os.path.join(postprocess_dir, 'enrichment_summary.json')
    with open(json_path, 'w') as f:
        json.dump(json_comps, f, indent=2, default=str)
    print(f"[wgcna_full] Wrote enrichment JSON to {json_path}")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .run_commands(
        "apt-get update && apt-get install -y git build-essential",
        "pip install --no-cache-dir numpy pandas scipy scikit-learn matplotlib tqdm "
        "networkx",
    )
    # PyTorch with CUDA for GPU steps
    .pip_install("torch", "torchvision", "torchaudio",
                 extra_index_url="https://download.pytorch.org/whl/cu124")
    .run_commands(
        "pip install --no-cache-dir pyg-lib torch-scatter torch-sparse torch-cluster "
        "torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.6.0+cu124.html",
    )
    # Project-specific deps (no pydeseq2 — we use log-CPM)
    .run_commands(
        "pip install --no-cache-dir vector-quantize-pytorch pycombat PyWGCNA",
    )
    # Copy the rol package into the image
    .add_local_dir(".", "/root/rol_repo", copy=True, ignore=[
        "./main.py",
        "./modal_pipeline.py",
        "./data/",
        "./output/",
        "./results_batch_corrected/",
        "./models/",
        "./.git",
        "./.gitignore",
        "./.idea",
        "./__pycache__",
        "./rol/__pycache__",
        "./vqgraph/__pycache__",
    ])
)

app = modal.App("rol-pipeline", image=image)

# ---------------------------------------------------------------------------
# CPU steps: 0 (batch correction) → 1 (matrices) → 2 (graphs)
# ---------------------------------------------------------------------------
@app.function(
    volumes={"/vol": volume},
    timeout=86400,      # 24 h — graph construction is slow
    gpu=None,            # CPU only
    memory=65536,        # 64 GB (graphs ~32 GB peak)
    cpu=8,
)
def cpu_steps(batch_n_genes: int = 10000, top_n_genes: int = None,
              min_expression: float = 0.0, run_id: str = None):
    import os
    import sys
    import subprocess

    volume.reload()
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]
    _acquire_run_lock(run_id)
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        repo = "/root/rol_repo"
        sys.path.insert(0, repo)
        os.chdir(repo)

        cmd = [
            sys.executable, "-u", os.path.join(repo, "pipeline.py"),
            # Data — read from volume
            "--transcriptome", os.path.join(DATA_DIR, "7.HKTranscriptome.csv"),
            "--metadata",      os.path.join(DATA_DIR, "1.Metadata.csv"),
            "--morphology",    os.path.join(DATA_DIR, "2.Morphology.csv"),
            "--infection",     os.path.join(DATA_DIR, "3.Infection.csv"),
            # Output — write to volume
            "--output-dir",    OUTPUT_DIR,
            # Batch correction
            "--batch-correct",
            "--batch-transcriptome", os.path.join(DATA_DIR, "7.HKTranscriptome.csv"),
            "--batch-n-genes", str(batch_n_genes),
            # Stop after graphs (step 2) — no GPU needed beyond this
            "--stop-after", "2",
            # "--force",
        ]
        # CRITICAL: forward the gene filter to the GRAPH-BUILD phase too.
        # Previously only train_steps passed --top-n-genes/--min-expression,
        # so graphs were built on ALL genes while the model was sized for
        # top_n_genes → node-count mismatch at training time.
        if top_n_genes is not None:
            cmd.extend(["--top-n-genes", str(top_n_genes)])
        if min_expression > 0:
            cmd.extend(["--min-expression", str(min_expression)])

        print(f"[cpu_steps] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[cpu_steps] Done — graphs and matrices are in {OUTPUT_DIR}")

        # Commit volume to persist checkpoints
        volume.commit()
    finally:
        _release_run_lock(run_id)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GPU training: 3 (train VQGNN) → 4 (embeddings) → 5 (mappings) → ...
# ---------------------------------------------------------------------------
@app.function(
    volumes={"/vol": volume},
    timeout=86400,
    gpu='H100',
    memory=65536,
    cpu=8,
)
def train_steps(codebook_size: int = 100, top_n_genes: int = None,
                min_expression: float = 0.0, train_joint: bool = True,
                train_both: bool = False, graph_batch_size: int = 0,
                resume: bool = False, epochs: int = 20,
                start_epoch: int = None, run_id: str = None):
    """Run pipeline steps 3-10 on GPU.

    Parameters
    ----------
    train_joint : bool
        Use joint (lake-conditioned) training.  Ignored when train_both=True.
    train_both : bool
        Train BOTH sequential and joint from a single graph load.
    graph_batch_size : int
        Load graphs in batches of this size (0 = all at once).
        Set to ~50 when training with all genes to limit CPU RAM.
    resume : bool
        Continue training from existing *_last.pt checkpoints instead of
        restarting.  Requires train_both + graph_batch_size (batched mode).
        ``epochs`` is the TOTAL target — training runs from the last
        completed epoch up to ``epochs``.
    epochs : int
        Total epoch count (cumulative).  When resuming, only the epochs
        beyond the last completed one are trained.
    start_epoch : int or None
        Override auto-detected completed_epochs when resuming.  Pass
        --start-epoch 20 to skip training entirely and only rerun steps 4-10.
    run_id : str
        Shared with cpu_steps for the same entrypoint invocation, so the
        serial cpu→train flow is treated as one run by the lock.
    """
    import os, sys, subprocess

    volume.reload()
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]
    _acquire_run_lock(run_id)
    try:
        if codebook_size == 100:
            out_dir = OUTPUT_DIR
        else:
            out_dir = os.path.join(OUTPUT_DIR, f"codes_{codebook_size}")
        os.makedirs(out_dir, exist_ok=True)

        repo = "/root/rol_repo"
        sys.path.insert(0, repo)
        os.chdir(repo)

        cmd = [
            sys.executable, "-u", os.path.join(repo, "pipeline.py"),
            "--graphs-dir", OUTPUT_DIR,
            "--device", "cuda",
            "--output-dir",    out_dir,
            "--epochs", str(epochs),
            "--lr", "1e-4",
            "--commit-alpha", "0.25",
            "--noise-scale", "5.0",
            "--codebook-size", str(codebook_size),
            "--label-permutations", "1000",
        ]
        if train_both:
            cmd.append("--train-both")
        elif train_joint:
            cmd.append("--train-joint")
        if top_n_genes is not None:
            cmd.extend(["--top-n-genes", str(top_n_genes)])
        if min_expression > 0:
            cmd.extend(["--min-expression", str(min_expression)])
        if graph_batch_size > 0:
            cmd.extend(["--graph-batch-size", str(graph_batch_size)])
        if resume:
            if not (train_both and graph_batch_size > 0):
                # Silently dropping --resume made `train_all --resume
                # --epochs 30` train nothing extra and silently reuse the old
                # checkpoint.  Fail loudly instead.
                raise ValueError(
                    "--resume requires --train-both AND --graph-batch-size > 0 "
                    f"(got train_both={train_both}, "
                    f"graph_batch_size={graph_batch_size}). Refusing to "
                    "silently ignore the resume request.")
            cmd.append("--resume")
        if start_epoch is not None:
            cmd.extend(["--start-epoch", str(start_epoch)])

        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        label = "both" if train_both else ("joint" if train_joint else "sequential")
        print(f"[train_{label}] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        print(f"[train_{label}] Done — outputs in {out_dir}")
        volume.commit()
    finally:
        _release_run_lock(run_id)


# ---------------------------------------------------------------------------
# Cleanup: remove cached model + downstream files before retraining
# ---------------------------------------------------------------------------
@app.function(
    volumes={"/vol": volume},
    timeout=600,
    gpu=None,
    memory=2048,
    cpu=1,
)
def cleanup(keep_joint: bool = False):
    """Remove cached model + downstream files before retraining.

    When ``keep_joint=True``, preserves ``*_joint.*`` files and
    ``figures_joint/`` (same as the old ``cleanup_sequential``).

    Refuses to run if another entrypoint is in progress (see the active-run
    lock) so it cannot destroy an in-flight run's partial graphs.
    """
    import os, shutil

    volume.reload()
    _check_no_active_run()

    if keep_joint:
        to_remove = ["model.pt", "model.pt.progress.json", "model_kwargs.pkl",
                     "embeddings.pkl", "gene_mappings.pkl",
                     "slopes.pkl", "clustering.pkl", "label_perm.pkl"]
        dirs_to_remove = ["figures", "postprocess"]
    else:
        to_remove = [
            "model.pt", "model_joint.pt",
            "model.pt.progress.json", "model_joint.pt.progress.json",
            "embeddings.pkl", "embeddings_joint.pkl",
            "gene_mappings.pkl", "gene_mappings_joint.pkl",
            "slopes.pkl", "slopes_joint.pkl",
            "clustering.pkl", "clustering_joint.pkl",
            "label_perm.pkl", "label_perm_joint.pkl",
        ]
        dirs_to_remove = ["figures", "figures_joint", "postprocess"]

    for fn in to_remove:
        path = os.path.join(OUTPUT_DIR, fn)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed {fn}")

    for fn in os.listdir(OUTPUT_DIR):
        if fn.endswith('.progress.json'):
            path = os.path.join(OUTPUT_DIR, fn)
            os.remove(path)
            print(f"  Removed {fn}")

    for dn in dirs_to_remove:
        path = os.path.join(OUTPUT_DIR, dn)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"  Removed {dn}/")

    label = " (sequential only)" if keep_joint else ""
    print(f"[cleanup{label}] Done — safe to retrain.")
    volume.commit()


# Backward-compat wrappers
@app.function(
    volumes={"/vol": volume}, timeout=600,
    gpu=None, memory=2048, cpu=1,
)
def cleanup_sequential():
    """Deprecated — use cleanup(keep_joint=True)."""
    cleanup(keep_joint=True)


@app.function(
    volumes={"/vol": volume}, timeout=86400,
    gpu='H100', memory=65536, cpu=4,
)
def gpu_steps(codebook_size: int = 100, top_n_genes: int = None,
              min_expression: float = 0.0):
    """Deprecated — use train_steps(train_joint=True)."""
    train_steps(codebook_size, top_n_genes, min_expression, train_joint=True)


@app.function(
    volumes={"/vol": volume}, timeout=86400,
    gpu='H100', memory=65536, cpu=8,
)
def sequential_train(codebook_size: int = 100, top_n_genes: int = None,
                     min_expression: float = 0.0):
    """Deprecated — use train_steps(train_joint=False)."""
    train_steps(codebook_size, top_n_genes, min_expression, train_joint=False)


# ---------------------------------------------------------------------------
# WGCNA baseline (runs on Modal CPU with PyWGCNA installed)
# ---------------------------------------------------------------------------
@app.function(
    volumes={"/vol": volume},
    timeout=14400,       # 4 hours
    gpu=None,
    memory=65536,
    cpu=4,
)
def wgcna_full(top_n_genes: int = 5000, min_expression: float = 1e-5,
               codebook_size: int = 100):
    """Comprehensive WGCNA: infection, sex, year comparisons with VQ overlap.

    Runs separate WGCNA per group, two-way module preservation, consensus
    module identification, eigengene → g:Profiler enrichment, and
    VQ-code vs WGCNA-module overlap analysis.

    ``codebook_size`` must match the training run: train_steps redirects its
    outputs to ``OUTPUT_DIR/codes_{N}`` when N != 100, so this function reads
    the model outputs (mappings, figures) from the same redirected dir rather
    than silently consuming stale root files.
    """
    import os, sys, subprocess

    repo = "/root/rol_repo"
    sys.path.insert(0, repo)
    os.chdir(repo)

    # Mirror train_steps' out_dir resolution so downstream reads match where
    # training wrote.  matrices.pkl stays in OUTPUT_DIR (the graph-build phase
    # writes it there regardless of codebook_size).
    out_dir = (OUTPUT_DIR if codebook_size == 100
               else os.path.join(OUTPUT_DIR, f"codes_{codebook_size}"))
    matrices_path = os.path.join(OUTPUT_DIR, "matrices.pkl")
    wgcna_dir = os.path.join(out_dir, "wgcna")
    script = os.path.join(repo, "scripts", "wgcna_pipeline.py")

    # Map paradigm to the correct gene_mappings file: sequential and joint
    # models produce different gene→code assignments, so gene indices in a
    # *_genes_for_go.csv must be resolved against the matching mappings file.
    def _gm_for_figures(fig_dir):
        if fig_dir and 'joint' in os.path.basename(
            os.path.normpath(fig_dir).rstrip('/')):
            return os.path.join(out_dir, "gene_mappings_joint.pkl")
        return os.path.join(out_dir, "gene_mappings.pkl")

    if not os.path.exists(matrices_path):
        print("ERROR: matrices.pkl not found — run cpu_steps first")
        return
    # Require at least one gene_mappings file to exist (in the out_dir that
    # matches the codebook_size the training used).
    if not (os.path.exists(os.path.join(out_dir, "gene_mappings.pkl"))
            or os.path.exists(os.path.join(out_dir, "gene_mappings_joint.pkl"))):
        print(f"ERROR: no gene_mappings.pkl found in {out_dir} — run steps 3-5 "
              f"first (with the matching --codebook-size)")
        return

    # VQ gene CSVs from pipeline step 9 (auto-detect joint vs sequential)
    vq_figures = os.path.join(out_dir, 'figures_joint')
    vq_paradigm = 'joint'
    if not os.path.isdir(vq_figures):
        vq_figures = os.path.join(out_dir, 'figures')
        vq_paradigm = 'sequential'
    alt_vq_figures = os.path.join(out_dir, 'figures') if vq_paradigm == 'joint' \
                     else os.path.join(out_dir, 'figures_joint')
    alt_paradigm = 'sequential' if vq_paradigm == 'joint' else 'joint'
    if not os.path.isdir(alt_vq_figures):
        alt_vq_figures = None

    cmd = [
        sys.executable, "-u", script,
        "--matrices", matrices_path,
        "--gene-mappings", _gm_for_figures(vq_figures),
        "--output-dir", wgcna_dir,
        "--top-n-genes", str(top_n_genes),
        "--min-expression", str(min_expression),
        "--loc-tsv", os.path.join(DATA_DIR, "gene_name_to_locid.tsv"),
        "--loc-cache", os.path.join(out_dir, "ncbi_loc_cache.json"),
        "--vq-figures-dir", vq_figures,
    ]
    print(f"[wgcna_full] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # --- VQ g:Profiler enrichment for role AND alt-paradigm codes ---
    # The WGCNA subprocess above uses --vq-figures-dir to pick ONE paradigm's
    # *_genes_for_go.csv files.  We also process:
    #   1. Role (Source/Recipient) — not in WGCNA's COMPARISONS list
    #   2. The ALT paradigm's CSVs — so manuscript_stats.json includes BOTH
    #      sequential (S) and joint (J) codes, differentiated by a _paradigm
    #      field ('sequential' / 'joint').
    vq_script = os.path.join(repo, "scripts", "vq_gprofiler.py")

    def _run_vq_enrichment(csv_path, out_dir, label):
        """Run vq_gprofiler.py on a single *_genes_for_go.csv."""
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, 'vq_code_enrichment.csv')
        cmd = [sys.executable, "-u", vq_script,
               "--codes", csv_path,
               "--gene-mappings", _gm_for_figures(os.path.dirname(csv_path)),
               "--output", out_csv,
               "--p-threshold", "0.05",
               "--loc-tsv", os.path.join(DATA_DIR, "gene_name_to_locid.tsv"),
               "--loc-cache", os.path.join(out_dir, "ncbi_loc_cache.json")]
        print(f"[wgcna_full] {label}: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            return True
        except Exception as exc:
            print(f"[wgcna_full] {label} failed: {exc}")
            return False

    # 1. Role — both paradigms (always tag with paradigm so
    #    _write_enrichment_summary correctly identifies sequential vs joint).
    for vq_dir, paradigm in [(vq_figures, vq_paradigm)] + \
                             ([(alt_vq_figures, alt_paradigm)]
                              if alt_vq_figures else []):
        role_csv = os.path.join(vq_dir, 'role_genes_for_go.csv')
        if os.path.exists(role_csv):
            out_dir = os.path.join(wgcna_dir, f'role_{paradigm}')
            _run_vq_enrichment(role_csv, out_dir,
                               f"Role enrichment ({paradigm})")

    # --- Write enrichment summary to postprocess ------------------------
    _write_enrichment_summary(wgcna_dir, os.path.join(out_dir, 'postprocess'))

    # --- Refresh manuscript_stats.json so the VQ/WGCNA g:Profiler tables
    #     and intersections (items 5-7) are current.  The step-10 copy was
    #     written before this phase produced enrichment_summary.json.
    ms_script = os.path.join(repo, "scripts", "extract_manuscript_stats.py")
    if os.path.exists(ms_script):
        ms_cmd = [sys.executable, "-u", ms_script, "--output-dir", out_dir]
        print(f"[wgcna_full] Refreshing manuscript_stats.json: {' '.join(ms_cmd)}")
        try:
            subprocess.run(ms_cmd, check=False)
        except Exception as exc:
            print(f"[wgcna_full] manuscript_stats refresh failed: {exc}")

    volume.commit()
    print("[wgcna_full] Done — results in wgcna/ and postprocess/enrichment_summary.md")


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------
# Regenerate gene mappings and embeddings from original model weights
# ---------------------------------------------------------------------------
def _regenerate_original_mappings_impl(train_joint: bool):
    """Shared implementation for regenerating original mappings."""
    import os, sys, pickle, torch

    repo = "/root/rol_repo"
    sys.path.insert(0, repo)
    os.chdir(repo)

    from surge.embedder import LakeEmbedder
    from surge.vqgnn import VQGNN
    from tqdm import tqdm

    sfx = "_joint" if train_joint else ""
    label = "joint" if train_joint else "sequential"

    # Load model kwargs
    kwargs_path = os.path.join(OUTPUT_DIR, f"model{sfx}_kwargs.pkl")
    with open(kwargs_path, "rb") as f:
        kwargs = pickle.load(f)
    print(f"[{label}] Model kwargs: {kwargs}")

    # Reconstruct model architecture
    print(f"[{label}] Building model architecture...")
    model = VQGNN(
        n_nodes=kwargs["n_nodes"],
        in_channels=kwargs["in_channels"],
        hidden_channels=kwargs["hidden_channels"],
        out_channels=kwargs["out_channels"],
        num_layers=kwargs["num_layers"],
        dropout=kwargs["dropout"],
        codebook_channels=kwargs["codebook_channels"],
        codebook_size=kwargs["codebook_size"],
        decoder_channels=kwargs["decoder_channels"],
        n_lakes=kwargs.get("n_lakes"),
    )

    # Load original state_dict
    model_filename = "model_joint_ORIGINAL.pt" if train_joint else "model_ORIGINAL.pt"
    model_path = os.path.join(OUTPUT_DIR, model_filename)
    print(f"[{label}] Loading original state_dict ({os.path.getsize(model_path)/1e6:.0f} MB)...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[{label}] Model loaded successfully.")

    # Use the manifest directly — embedder loads graphs on-the-fly from paths.
    # Wrap in a thin tqdm adapter so we see progress without changing embedder code.
    manifest_path = os.path.join(OUTPUT_DIR, "graphs_manifest.pkl")
    with open(manifest_path, "rb") as f:
        graphs_manifest = pickle.load(f)
    print(f"[{label}] Loaded manifest with {len(graphs_manifest)} keys")

    class _TqdmDict:
        """Dict wrapper that adds a progress bar to .items() iteration."""
        def __init__(self, d, desc):
            self._d = d
            self._desc = desc
        def items(self):
            return tqdm(self._d.items(), desc=self._desc, unit="key")

    embedder = LakeEmbedder(model, device="cpu")

    # Regenerate gene mappings only (embeddings on Modal are still original).
    # A joint model must be conditioned on its lake indices at eval time (the
    # same mapping the training path builds), or build_gene_vq_mappings would
    # raise for the joint model.
    lake_name_to_id = None
    if train_joint:
        lake_names = sorted(
            set(str(k).split(' (')[0] for k in graphs_manifest))
        lake_name_to_id = {name: i for i, name in enumerate(lake_names)}
    print(f"[{label}] Building gene-VQ mappings from original model...")
    vq_to_gene, gene_to_vq = embedder.build_gene_vq_mappings(
        _TqdmDict(graphs_manifest, f"[{label}] Gene mappings"),
        lake_name_to_id=lake_name_to_id)

    # Get gene names from existing mappings (they don't change with retraining)
    existing_gm_path = os.path.join(OUTPUT_DIR, f"gene_mappings{sfx}.pkl")
    gene_names = None
    if os.path.exists(existing_gm_path):
        with open(existing_gm_path, "rb") as f:
            existing_gm = pickle.load(f)
        gene_names = existing_gm.get("gene_names")

    mappings = {
        "vq_to_gene": vq_to_gene,
        "gene_to_vq": gene_to_vq,
        "gene_names": gene_names,
    }
    gm_path = os.path.join(OUTPUT_DIR, f"gene_mappings{sfx}_ORIGINAL.pkl")
    with open(gm_path, "wb") as f:
        pickle.dump(mappings, f)
    print(f"[{label}] Saved gene mappings to {gm_path}")

    volume.commit()
    print(f"[{label}] Done.")


@app.function(
    volumes={"/vol": volume},
    timeout=3600,
    memory=32768,
)
def regenerate_original_mappings_joint():
    """Regenerate gene mappings + embeddings from the original joint model."""
    _regenerate_original_mappings_impl(train_joint=True)


@app.function(
    volumes={"/vol": volume},
    timeout=3600,
    memory=32768,
)
def regenerate_original_mappings_sequential():
    """Regenerate gene mappings + embeddings from the original sequential model."""
    _regenerate_original_mappings_impl(train_joint=False)


@app.local_entrypoint()
def run_original_mappings():
    """Upload original models and regenerate original mappings for both paradigms."""
    import subprocess, os

    # Upload both models.  IMPORTANT: `modal volume put` paths are relative to
    # the volume root, and _regenerate_original_mappings_impl reads from
    # OUTPUT_DIR = /vol/25k_v5.  The old targets ('output/...') landed the
    # files at /vol/output/... — a different directory — so the regenerate step
    # always hit FileNotFoundError.  Use the 25k_v5/ prefix to match.
    for local, remote in [
        ("output/model_joint.pt",
         "25k_v5/model_joint_ORIGINAL.pt"),
        ("output/model.pt",
         "25k_v5/model_ORIGINAL.pt"),
    ]:
        print(f"Uploading {local} -> {remote}...")
        subprocess.run(
            ["modal", "volume", "put", "rol_output", local, remote],
            check=True,
        )

    # Regenerate joint
    print("\n=== Regenerating JOINT mappings ===")
    regenerate_original_mappings_joint.remote()

    # Regenerate sequential
    print("\n=== Regenerating SEQUENTIAL mappings ===")
    regenerate_original_mappings_sequential.remote()

    print("\nDone. Pull with: modal volume get rol_output 25k_v5/gene_mappings_ORIGINAL.pkl ...")


@app.function(
    volumes={"/vol": volume},
    timeout=86400,
    gpu=None,
    memory=65536,
    cpu=8,
)
def regenerate_figures_remote():
    """Re-run step 9 (figures) for both joint and sequential paradigms.

    Uses cached model weights, embeddings, clustering, etc. from the volume.
    No retraining or recomputation of steps 0-8.
    """
    import os, sys, subprocess

    repo = "/root/rol_repo"
    sys.path.insert(0, repo)
    os.chdir(repo)

    for train_flag, label in [("--train-joint", "joint"), ("", "sequential")]:
        cmd = [
            sys.executable, "-u", os.path.join(repo, "pipeline.py"),
            "--graphs-dir", OUTPUT_DIR,
            "--output-dir", OUTPUT_DIR,
            "--device", "cpu",
            "--epochs", "20",
            "--lr", "1e-4",
            "--commit-alpha", "0.25",
            "--noise-scale", "5.0",
            "--codebook-size", "100",
            "--label-permutations", "1000",
            "--stop-after", "10",
        ]
        if train_flag:
            cmd.append(train_flag)

        print(f"[regenerate_figures] Running steps 9-10 ({label})...")
        subprocess.run(cmd, check=True)
        print(f"[regenerate_figures] Done ({label})")

    volume.commit()


@app.local_entrypoint()
def regenerate_figures():
    """Re-run step 9 (figures) for both joint and sequential paradigms.

    Usage:
        modal run modal_pipeline.py::regenerate_figures
    """
    regenerate_figures_remote.remote()
    print("=== regenerate_figures complete ===")


# ---------------------------------------------------------------------------
# Main entrypoints — two commands for end-to-end runs
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def joint(codebook_size: int = 100, top_n_genes: int = None,
          min_expression: float = 0.0, batch_n_genes: int = 10000):
    """End-to-end joint (lake-conditioned) training pipeline.

    Phase 1 — CPU: steps 0-2 (batch correction, expression matrices, graphs)
    Phase 2 — GPU: steps 3-10 (train VQGNN with lake conditioning, figures,
              post-processing)

    Usage:
        modal run modal_pipeline.py::joint
        modal run modal_pipeline.py::joint --codebook-size 50
        modal run modal_pipeline.py::joint --batch-n-genes 0  # all genes
    """
    print("=== Phase 1: CPU (steps 0-2) ===")
    run_id = uuid.uuid4().hex[:8]
    cpu_steps.remote(batch_n_genes, top_n_genes, min_expression,
                     run_id=run_id)
    print("=== Phase 2: GPU joint training (steps 3-10) ===")
    train_steps.remote(codebook_size, top_n_genes, min_expression,
                       train_joint=True, run_id=run_id)
    print("=== joint complete ===")


@app.local_entrypoint()
def sequential(codebook_size: int = 100, top_n_genes: int = None,
               min_expression: float = 0.0, batch_n_genes: int = 10000):
    """End-to-end sequential (per-graph) training pipeline.

    Phase 1 — CPU: steps 0-2 (batch correction, expression matrices, graphs)
    Phase 2 — GPU: steps 3-10 (train VQGNN per-graph, figures, post-processing)

    Usage:
        modal run modal_pipeline.py::sequential
        modal run modal_pipeline.py::sequential --batch-n-genes 0  # all genes
    """
    print("=== Phase 1: CPU (steps 0-2) ===")
    run_id = uuid.uuid4().hex[:8]
    cpu_steps.remote(batch_n_genes, top_n_genes, min_expression,
                     run_id=run_id)
    print("=== Phase 2: GPU sequential training (steps 3-10) ===")
    train_steps.remote(codebook_size, top_n_genes, min_expression,
                       train_joint=False, run_id=run_id)
    print("=== sequential complete ===")


@app.local_entrypoint()
def train_all(codebook_size: int = 100, top_n_genes: int = None,
              min_expression: float = 0.0, batch_n_genes: int = 10000,
              graph_batch_size: int = 0, resume: bool = False,
              epochs: int = 20, start_epoch: int = None):
    """Train BOTH sequential and joint in one run — graphs loaded once.

    Phase 1 — CPU: steps 0-2 (batch correction, expression matrices, graphs)
    Phase 2 — GPU: steps 3-10 for BOTH paradigms from a single graph load
    Phase 3 — CPU: WGCNA baseline (infection, sex, year comparisons + VQ overlap)

    Usage:
        modal run modal_pipeline.py::train_all
        modal run modal_pipeline.py::train_all --batch-n-genes 0  # all genes
        modal run modal_pipeline.py::train_all --batch-n-genes 0 --graph-batch-size 50

    Resume (continue training from an existing checkpoint, skipping the CPU
    graph-building phase):
        modal run modal_pipeline.py::train_all --resume --epochs 30 \\
            --graph-batch-size 50

    Skip training entirely — only rerun downstream steps 4-10 on the
    existing model:
        modal run modal_pipeline.py::train_all --graph-batch-size 75 --resume \\
            --start-epoch 20
    """
    run_id = uuid.uuid4().hex[:8]
    if resume:
        print("=== --resume: skipping CPU phase, continuing training ===")
    else:
        print("=== Phase 1: CPU (steps 0-2) ===")
        cpu_steps.remote(batch_n_genes, top_n_genes, min_expression,
                         run_id=run_id)
    print("=== Phase 2: GPU train-both (steps 3-10 × 2 paradigms) ===")
    train_steps.remote(codebook_size, top_n_genes, min_expression,
                       train_both=True, graph_batch_size=graph_batch_size,
                       resume=resume, epochs=epochs, start_epoch=start_epoch,
                       run_id=run_id)
    print("=== Phase 3: WGCNA baseline ===")
    wgcna_full.remote(codebook_size=codebook_size)
    print("=== train_all complete ===")


@app.function(
    volumes={"/vol": volume},
    timeout=3600,
    gpu=None,
    memory=16384,
    cpu=4,
)
def postprocess_remote():
    """Re-run step 10 (post-processing) only — requires existing outputs."""
    import os, sys

    repo = "/root/rol_repo"
    sys.path.insert(0, repo)
    os.chdir(repo)

    # Minimal args object — only output_dir is needed by step_postprocess
    class Args:
        output_dir = OUTPUT_DIR

    from pipeline import step_postprocess
    step_postprocess(Args())
    print("[postprocess] Done")
    volume.commit()


@app.local_entrypoint()
def postprocess():
    """Re-run step 10 (post-processing) only — requires existing outputs.

    Usage:
        modal run modal_pipeline.py::postprocess
    """
    postprocess_remote.remote()
    print("=== postprocess complete ===")


