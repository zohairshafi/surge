#!/usr/bin/env python3
"""
Simulation sanity check: VQGNN recovery of known network structures.

Generates 1000 genes across 10 different co-expression network structures
(varying module count and coupling strength), trains VQGNN with sequential
training (graph-order shuffled each epoch), and evaluates whether embeddings
cluster by network.

Usage:
    python scripts/simulation_sanity_check.py --seed 42
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
import torch
import torch.optim as optim
from torch.cuda.amp import autocast
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add project root to path so we can import rol modules
# ---------------------------------------------------------------------------
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


# ===========================================================================
# 1. Network definitions
# ===========================================================================

def define_networks(n_genes=1000, seed=42):
    """Return list of 10 network config dicts.

    Each dict has:
        name : str
        membership : np.ndarray (n_genes,) — module index per gene
        beta : float or 'hierarchical'
        description : str
        n_modules : int
        strength : str ('strong', 'weak', 'very_weak', 'hierarchical')
    """
    rng = np.random.default_rng(seed)
    networks = []

    # Helper: assign genes to modules given a list of sizes
    def assign_modules(sizes):
        membership = np.empty(n_genes, dtype=np.int32)
        start = 0
        for mod_idx, sz in enumerate(sizes):
            membership[start:start + sz] = mod_idx
            start += sz
        # Shuffle to randomize which genes are in which module
        perm = rng.permutation(n_genes)
        return membership[perm.argsort()]  # un-shuffle so genes 0..n-1 aren't sorted

    # ---- Network 0: 2 large modules, strong ----
    networks.append({
        "name": "2mod_strong",
        "membership": assign_modules([500, 500]),
        "beta": 0.9,
        "strength": "strong",
        "n_modules": 2,
        "description": "2 large modules (500+500), strong coupling (beta=0.9)",
    })

    # ---- Network 1: 2 large modules, weak ----
    networks.append({
        "name": "2mod_weak",
        "membership": assign_modules([500, 500]),
        "beta": 0.4,
        "strength": "weak",
        "n_modules": 2,
        "description": "2 large modules (500+500), weak coupling (beta=0.4)",
    })

    # ---- Network 2: 5 medium modules, strong ----
    networks.append({
        "name": "5mod_strong",
        "membership": assign_modules([200] * 5),
        "beta": 0.9,
        "strength": "strong",
        "n_modules": 5,
        "description": "5 medium modules (200x5), strong coupling (beta=0.9)",
    })

    # ---- Network 3: 5 medium modules, weak ----
    networks.append({
        "name": "5mod_weak",
        "membership": assign_modules([200] * 5),
        "beta": 0.4,
        "strength": "weak",
        "n_modules": 5,
        "description": "5 medium modules (200x5), weak coupling (beta=0.4)",
    })

    # ---- Network 4: 5 medium modules, very weak ----
    networks.append({
        "name": "5mod_vweak",
        "membership": assign_modules([200] * 5),
        "beta": 0.2,
        "strength": "very_weak",
        "n_modules": 5,
        "description": "5 medium modules (200x5), very weak coupling (beta=0.2)",
    })

    # ---- Network 5: 10 small modules, strong ----
    networks.append({
        "name": "10mod_strong",
        "membership": assign_modules([100] * 10),
        "beta": 0.9,
        "strength": "strong",
        "n_modules": 10,
        "description": "10 small modules (100x10), strong coupling (beta=0.9)",
    })

    # ---- Network 6: 25 tiny modules, strong ----
    networks.append({
        "name": "25mod_strong",
        "membership": assign_modules([40] * 25),
        "beta": 0.9,
        "strength": "strong",
        "n_modules": 25,
        "description": "25 tiny modules (40x25), strong coupling (beta=0.9)",
    })

    # ---- Network 7: 50 tiny modules, strong ----
    networks.append({
        "name": "50mod_strong",
        "membership": assign_modules([20] * 50),
        "beta": 0.9,
        "strength": "strong",
        "n_modules": 50,
        "description": "50 tiny modules (20x50), strong coupling (beta=0.9)",
    })

    # ---- Network 8: 5 unequal modules, strong ----
    networks.append({
        "name": "5mod_unequal",
        "membership": assign_modules([400, 250, 150, 100, 100]),
        "beta": 0.9,
        "strength": "strong",
        "n_modules": 5,
        "description": "5 unequal modules (400,250,150,100,100), strong (beta=0.9)",
    })

    # ---- Network 9: Hierarchical (2 super x 3 sub) ----
    mem = np.empty(n_genes, dtype=np.int32)
    super_sizes = [500, 500]
    sub_per_super = 3
    sub_size = [167, 167, 166]  # ~500/3 per super-module
    gene_idx = 0
    sub_idx = 0
    for super_id, n_super in enumerate(super_sizes):
        for sub_id in range(sub_per_super):
            sz = sub_size[sub_id]
            for _ in range(sz):
                if gene_idx < n_genes:
                    mem[gene_idx] = super_id * sub_per_super + sub_id
                    gene_idx += 1
    # Shuffle
    perm = rng.permutation(n_genes)
    mem = mem[perm.argsort()]
    networks.append({
        "name": "hierarchical",
        "membership": mem,
        "beta": "hierarchical",
        "strength": "hierarchical",
        "n_modules": 6,
        "beta_super": 0.3,
        "beta_sub": 0.8,
        "super_module": np.array([
            0 if m < 3 else 1 for m in mem
        ]),
        "description": "Hierarchical: 2 super-modules, 3 sub-modules each (beta_super=0.3, beta_sub=0.8)",
    })

    return networks


# ===========================================================================
# 2. Expression data generation
# ===========================================================================

def generate_expression(network, n_fish, rng):
    """Generate (n_fish, n_genes) expression matrix from a factor model.

    Flat networks:  x_j = beta * f_module(j) + eps,  eps ~ N(0, 1-beta^2)
    Hierarchical:   x_j = beta_super * f_super(j) + beta_sub * f_sub(j) + eps
                     eps ~ N(0, 1 - beta_super^2 - beta_sub^2)

    After generating, rows are normalized to sum to 1 (softmax convention)
    matching the pipeline's relative-abundance normalization.
    """
    n_genes = len(network["membership"])

    if network["beta"] == "hierarchical":
        beta_super = network["beta_super"]
        beta_sub = network["beta_sub"]
        noise_var = 1.0 - beta_super ** 2 - beta_sub ** 2
        if noise_var <= 0:
            noise_var = 0.01
        noise_std = np.sqrt(noise_var)

        n_super = 2
        n_sub = 6
        f_super = rng.normal(0, 1, n_super)       # (n_super,)
        f_sub = rng.normal(0, 1, n_sub)             # (n_sub,)

        X = np.empty((n_fish, n_genes), dtype=np.float64)
        for i in range(n_fish):
            f_s = rng.normal(0, 1, n_super)
            f_b = rng.normal(0, 1, n_sub)
            eps = rng.normal(0, noise_std, n_genes)
            for j in range(n_genes):
                sup = network["super_module"][j]
                sub = network["membership"][j]
                X[i, j] = (beta_super * f_s[sup] +
                           beta_sub * f_b[sub] +
                           eps[j])
    else:
        beta = network["beta"]
        noise_var = 1.0 - beta ** 2
        if noise_var <= 0:
            noise_var = 0.01
        noise_std = np.sqrt(noise_var)

        n_modules = network["n_modules"]
        membership = network["membership"]

        X = np.empty((n_fish, n_genes), dtype=np.float64)
        for i in range(n_fish):
            f = rng.normal(0, 1, n_modules)
            eps = rng.normal(0, noise_std, n_genes)
            X[i, :] = beta * f[membership] + eps

    # Column-wise z-score: preserve correlation structure directly.
    # Row-normalization (softmax or division) introduces compositional
    # coupling that creates dense background correlations (all graphs
    # end up 36-42% density regardless of module structure).  Z-scoring
    # per gene across fish preserves within-module correlations while
    # between-module correlations stay near zero.
    X_centered = X - X.mean(axis=0, keepdims=True)
    X_std = X_centered.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    X_z = X_centered / X_std

    return X_z.astype(np.float32)


# ===========================================================================
# 3. Build matrices dict
# ===========================================================================

def build_all_matrices(networks, n_fish=25, n_replicates=5, seed=42):
    """Generate all expression matrices and return (matrix_dict, labels_dict).

    matrix_dict: {"Net{id}_Rep{rep} (2019)": np.ndarray (n_fish, n_genes)}
    labels_dict: same keys -> network_id (0-9)
    """
    rng = np.random.default_rng(seed)
    matrix_dict = {}
    labels_dict = {}

    for net_id, net in enumerate(networks):
        for rep in range(n_replicates):
            key = f"Net{net_id}_Rep{rep} (2019)"
            # Each replicate gets its own sub-seed for independence
            sub_seed = seed * 1000 + net_id * 100 + rep
            rep_rng = np.random.default_rng(sub_seed)
            X = generate_expression(net, n_fish, rep_rng)
            matrix_dict[key] = X
            labels_dict[key] = net_id

    return matrix_dict, labels_dict


# ===========================================================================
# 4. Graph construction
# ===========================================================================

def build_graphs(matrix_dict):
    """Build co-expression graphs using CoexpressionGraphBuilder."""
    from surge.graphs import CoexpressionGraphBuilder

    builder = CoexpressionGraphBuilder(
        n_eigencomponents=33,
        reconstruction_levels=[2, 4, 16, 32],
        device="cpu",
    )
    graphs, radii_dict = builder.build_all(
        matrix_dict,
        output_dir=None,
        keep_in_memory=True,
        reuse_existing=False,
    )
    return graphs, radii_dict


# ===========================================================================
# 5. Training (sequential with per-epoch shuffling)
# ===========================================================================

def train_sequential_shuffled(model, graphs, radii_dict, output_dir,
                               epochs=5, lr=1e-4, commit_alpha=0.25,
                               device="cpu"):
    """Sequential training with graph order shuffled each epoch.

    Unlike train_joint, no lake-specific conditioning — the same gene always
    maps to the same VQ code regardless of graph context.
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = (device.type == "cuda")

    # Collect valid (non-degenerate) graph keys and their data
    all_keys = sorted(graphs.keys())
    valid_keys = []
    for key in all_keys:
        entry = graphs[key]
        if isinstance(entry, dict) and entry.get("degenerate"):
            continue
        valid_keys.append(key)

    print(f"  Training on {len(valid_keys)}/{len(all_keys)} valid graphs "
          f"({len(all_keys) - len(valid_keys)} degenerate)")

    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    loss_history = []

    for epoch in range(epochs):
        # Shuffle graph order
        perm = np.random.permutation(len(valid_keys))
        shuffled_keys = [valid_keys[i] for i in perm]

        epoch_edge_loss = 0.0
        epoch_commit_loss = 0.0
        n_graphs = 0
        # Per-epoch codebook-utilization accumulator (nearly free — the
        # forward pass already computes indices, we just stop discarding them).
        usage = torch.zeros(model.vq.codebook_size, dtype=torch.long)

        pbar = tqdm(shuffled_keys, desc=f"  Epoch {epoch+1}/{epochs}",
                     unit="graph")
        for key in pbar:
            g = graphs[key][0]  # first reconstruction level (k=2)
            r = radii_dict.get(key)

            edge_index = g.edge_index.to(device)
            target_adj = g.target_adj.to(device)

            if r is not None:
                r_tensor = torch.as_tensor(r, dtype=torch.float32, device=device)
            else:
                r_tensor = None

            optimizer.zero_grad()

            if use_amp:
                with autocast(dtype=torch.bfloat16):
                    _, decoded, _, indices, commit_loss = model(
                        edge_index, radii=r_tensor, lake_idx=None
                    )
                    edge_loss = model.reconstruction_loss(decoded, target_adj)
                    loss = edge_loss + commit_alpha * commit_loss
                usage += model.codebook_usage(indices)
            else:
                _, decoded, _, indices, commit_loss = model(
                    edge_index, radii=r_tensor, lake_idx=None
                )
                edge_loss = model.reconstruction_loss(decoded, target_adj)
                loss = edge_loss + commit_alpha * commit_loss
                usage += model.codebook_usage(indices)

            loss.backward()
            optimizer.step()

            el = edge_loss.item()
            cl = commit_loss.item()
            epoch_edge_loss += el
            epoch_commit_loss += cl
            n_graphs += 1
            pbar.set_postfix({"edge": f"{el:.4f}", "commit": f"{cl:.4f}"})

        avg_edge = epoch_edge_loss / n_graphs
        avg_commit = epoch_commit_loss / n_graphs
        total_loss = avg_edge + 0.1 * avg_commit
        print(f"    Epoch {epoch+1}: edge_loss={avg_edge:.4f}, "
              f"commit_loss={avg_commit:.4f}, crit={total_loss:.4f}")
        print("    " + model.format_codebook_usage(usage, 'sim'))

        loss_history.append({
            "epoch": epoch + 1,
            "edge_loss": avg_edge,
            "commit_loss": avg_commit,
            "total_loss": total_loss,
        })

        if total_loss < best_loss:
            best_loss = total_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    # Load best epoch
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Loaded best model from epoch {best_epoch} (loss={best_loss:.4f})")

    # Save
    model_path = os.path.join(output_dir, "simulation_model.pt")
    torch.save({"state_dict": model.state_dict(), "best_epoch": best_epoch,
                "best_loss": best_loss}, model_path)
    loss_path = os.path.join(output_dir, "simulation_model_loss.pkl")
    import pickle
    with open(loss_path, "wb") as f:
        pickle.dump({"loss_history": loss_history, "best_epoch": best_epoch}, f)

    return model


# ===========================================================================
# 6. Embedding extraction
# ===========================================================================

def extract_embeddings(model, graphs, device="cpu"):
    """Extract codebook histogram embeddings for all graphs."""
    model.eval()
    model.to(device)
    embeddings = {}
    degenerate_keys = []
    codebook_size = model.vq.codebook_size

    for key in sorted(graphs.keys()):
        entry = graphs[key]
        if isinstance(entry, dict) and entry.get("degenerate"):
            embeddings[key] = np.ones(codebook_size) / codebook_size
            degenerate_keys.append(key)
            continue

        g = entry[0]  # first reconstruction level
        edge_index = g.edge_index.to(device)

        with torch.no_grad():
            hist = model.get_codebook_histogram(edge_index, lake_idx=None)
        embeddings[key] = hist.cpu().numpy()

    if degenerate_keys:
        print(f"  {len(degenerate_keys)} degenerate graphs assigned uniform embedding")

    return embeddings


# ===========================================================================
# 7. Evaluation
# ===========================================================================

def evaluate(embeddings, labels_dict, networks, output_dir, seed=42):
    """Run all evaluation metrics and produce figures."""
    from surge.analysis import LakeAnalyzer
    from sklearn.decomposition import PCA

    analyzer = LakeAnalyzer(embeddings, data=None)
    all_keys = sorted(embeddings.keys())
    n_keys = len(all_keys)
    codebook_size = embeddings[all_keys[0]].shape[0]
    n_networks = len(networks)
    rng = np.random.default_rng(seed)

    # Extract metadata per key
    network_labels = {k: labels_dict[k] for k in all_keys}
    network_ids = np.array([labels_dict[k] for k in all_keys])

    metadata = []
    for k in all_keys:
        net_id = labels_dict[k]
        net = networks[net_id]
        metadata.append({
            "network_id": int(net_id),
            "n_modules": int(net["n_modules"]),
            "strength": str(net["strength"]),
        })

    metrics = {
        "n_genes": 1000,
        "n_networks": n_networks,
        "n_replicates": 5,
        "n_fish_per_replicate": 25,
        "n_graphs_total": n_keys,
        "codebook_size": codebook_size,
    }

    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    # ---- 7a. Silhouette score ----
    sil = analyzer.silhouette(network_labels)
    metrics["silhouette_network"] = float(sil) if not np.isnan(sil) else None
    print(f"\nSilhouette score (network labels): {sil:.4f}")

    # ---- 7b. Label permutation test ----
    # Manual implementation since label_permutation_test requires a built-in
    # label_type that build_labels() understands
    n_perm = 1000
    observed_sil = sil
    null_sils = []
    for _ in range(n_perm):
        permuted = dict(zip(all_keys, rng.permutation(network_ids)))
        null_sil = analyzer.silhouette(permuted)
        null_sils.append(null_sil if not np.isnan(null_sil) else 0.0)
    null_sils = np.array(null_sils)
    p_value = (np.sum(null_sils >= observed_sil) + 1) / (n_perm + 1)
    metrics["silhouette_permutation_p"] = float(p_value)
    metrics["silhouette_null_mean"] = float(null_sils.mean())
    metrics["silhouette_null_std"] = float(null_sils.std())
    print(f"Label permutation test: p={p_value:.4f} "
          f"(null mean={null_sils.mean():.4f}, std={null_sils.std():.4f})")

    # ---- 7c. PERMANOVA ----
    permanova = analyzer.permanova_decomposition(metadata, n_permutations=1000,
                                                  random_seed=seed)
    metrics["permanova"] = {}
    print("\nPERMANOVA decomposition:")
    for factor, result in permanova.items():
        print(f"  {factor}: R^2={result['r2']:.4f}, p={result['p_value']:.4f}")
        metrics["permanova"][factor] = {
            "r2": float(result["r2"]),
            "p_value": float(result["p_value"]),
        }

    # ---- 7d. Active codes and codebook statistics ----
    emb_matrix = np.vstack([embeddings[k] for k in all_keys])  # (n_graphs, K)
    code_mean_usage = emb_matrix.mean(axis=0)
    active_codes = int(np.sum(code_mean_usage > 0))
    # Normalized entropy: 0 = all mass on one code, 1 = uniform
    eps = 1e-12
    entropy = -np.sum(code_mean_usage * np.log(code_mean_usage + eps))
    max_entropy = np.log(codebook_size)
    norm_entropy = entropy / max_entropy
    metrics["active_codes"] = active_codes
    metrics["code_usage_entropy"] = float(entropy)
    metrics["code_usage_norm_entropy"] = float(norm_entropy)
    print(f"\nCodebook usage: {active_codes}/{codebook_size} active codes, "
          f"norm_entropy={norm_entropy:.4f}")

    # Per-network mean code distributions
    network_mean_hists = {}
    for net_id in range(n_networks):
        net_keys = [k for k in all_keys if labels_dict[k] == net_id]
        net_emb = np.vstack([embeddings[k] for k in net_keys])
        network_mean_hists[net_id] = net_emb.mean(axis=0)

    # Pairwise Jensen-Shannon divergence
    jsd_matrix = np.zeros((n_networks, n_networks))
    for i in range(n_networks):
        for j in range(n_networks):
            jsd_matrix[i, j] = float(jensenshannon(
                network_mean_hists[i], network_mean_hists[j]
            ))
    metrics["pairwise_jsd_mean"] = float(jsd_matrix[np.triu_indices(n_networks, 1)].mean())
    metrics["pairwise_jsd_range"] = [float(jsd_matrix.min()), float(jsd_matrix.max())]
    print(f"Pairwise JS divergence: mean={jsd_matrix.mean():.4f}, "
          f"range=[{jsd_matrix.min():.4f}, {jsd_matrix.max():.4f}]")

    # ---- 7e. PCA ----
    X_stack = np.vstack([embeddings[k] for k in all_keys])
    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(X_stack)
    metrics["pca_explained_variance"] = [
        float(pca.explained_variance_ratio_[0]),
        float(pca.explained_variance_ratio_[1]),
    ]

    # ---- 7f. Save metrics JSON ----
    metrics_path = os.path.join(output_dir, "simulation_metrics.json")
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    with open(metrics_path, "w") as f:
        json.dump(convert(metrics), f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # ---- 7g. Plots ----
    make_plots(
        all_keys=all_keys,
        projected=projected,
        network_ids=network_ids,
        networks=networks,
        null_sils=null_sils,
        observed_sil=observed_sil,
        p_value=p_value,
        permanova=permanova,
        jsd_matrix=jsd_matrix,
        network_mean_hists=network_mean_hists,
        code_mean_usage=code_mean_usage,
        codebook_size=codebook_size,
        output_dir=output_dir,
    )

    return metrics


def make_plots(all_keys, projected, network_ids, networks, null_sils,
               observed_sil, p_value, permanova, jsd_matrix,
               network_mean_hists, code_mean_usage, codebook_size,
               output_dir):
    """Generate all evaluation figures."""
    n_networks = len(networks)
    colors = plt.cm.tab10(np.linspace(0, 1, n_networks))

    # ---- Plot A: PCA colored by network ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    ax = axes[0, 0]
    for net_id in range(n_networks):
        mask = network_ids == net_id
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   c=[colors[net_id]], label=networks[net_id]["name"],
                   s=60, edgecolors="k", linewidths=0.3, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("(A) PCA of VQ code embeddings, colored by network")
    ax.legend(fontsize=6, ncol=2)

    # ---- Plot B: PCA colored by strength ----
    ax = axes[0, 1]
    strength_colors = {"strong": "#d62728", "weak": "#ff7f0e",
                       "very_weak": "#bcbd22", "hierarchical": "#9467bd"}
    for net_id in range(n_networks):
        mask = network_ids == net_id
        s = networks[net_id]["strength"]
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   c=[strength_colors[s]], label=f"{networks[net_id]['name']} ({s})",
                   s=60, edgecolors="k", linewidths=0.3, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("(B) PCA colored by coupling strength")
    ax.legend(fontsize=5.5, ncol=2)

    # ---- Plot C: Silhouette permutation null distribution ----
    ax = axes[0, 2]
    ax.hist(null_sils, bins=30, color="lightgray", edgecolor="gray", alpha=0.7)
    ax.axvline(observed_sil, color="red", linestyle="--", linewidth=2,
               label=f"Observed = {observed_sil:.3f}\np = {p_value:.4f}")
    ax.set_xlabel("Silhouette score")
    ax.set_ylabel("Frequency")
    ax.set_title("(C) Label permutation test (network labels)")
    ax.legend(fontsize=8)

    # ---- Plot D: PERMANOVA bar chart ----
    ax = axes[1, 0]
    factors = list(permanova.keys())
    r2_vals = [permanova[f]["r2"] for f in factors]
    p_vals = [permanova[f]["p_value"] for f in factors]
    bars = ax.bar(range(len(factors)), r2_vals, color="steelblue", edgecolor="k")
    # Significance stars
    for i, (f, p) in enumerate(zip(factors, p_vals)):
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        ax.text(i, r2_vals[i] + 0.005, stars, ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(factors, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Marginal R^2")
    ax.set_title("(D) PERMANOVA: variance explained by each factor")
    ax.set_ylim(0, max(r2_vals) * 1.25 if r2_vals else 1)

    # ---- Plot E: Pairwise JS divergence heatmap ----
    ax = axes[1, 1]
    im = ax.imshow(jsd_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_networks))
    ax.set_yticks(range(n_networks))
    ax.set_xticklabels([net["name"] for net in networks], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([net["name"] for net in networks], fontsize=7)
    ax.set_title("(E) Jensen-Shannon divergence between network mean code distributions")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ---- Plot F: Per-code variance across networks ----
    ax = axes[1, 2]
    # Compute variance of mean code usage across networks
    all_means = np.vstack([network_mean_hists[i] for i in range(n_networks)])
    code_var = all_means.var(axis=0)
    top_n = min(20, codebook_size)
    top_codes = np.argsort(code_var)[::-1][:top_n]
    ax.bar(range(top_n), code_var[top_codes], color="steelblue", edgecolor="k")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([str(c) for c in top_codes], rotation=90, fontsize=7)
    ax.set_xlabel("VQ code")
    ax.set_ylabel("Variance across networks")
    ax.set_title(f"(F) Top {top_n} most differentially-used VQ codes")

    fig.suptitle("VQGNN Simulation Sanity Check: Recovery of Known Network Structures",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "evaluation_summary.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved evaluation_summary.png")

    # ---- Extra: Network-mean code usage heatmap ----
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    # Show top 30 most variable codes
    top_codes_30 = np.argsort(code_var)[::-1][:30]
    heatmap_data = np.zeros((n_networks, len(top_codes_30)))
    for net_id in range(n_networks):
        heatmap_data[net_id, :] = network_mean_hists[net_id][top_codes_30]
    im2 = ax2.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(top_codes_30)))
    ax2.set_xticklabels([str(c) for c in top_codes_30], rotation=90, fontsize=6)
    ax2.set_yticks(range(n_networks))
    ax2.set_yticklabels(
        [f"{net['name']} ({net['strength']})" for net in networks], fontsize=8
    )
    ax2.set_xlabel("VQ code")
    ax2.set_ylabel("Network")
    ax2.set_title("Mean code usage per network (top 30 most variable codes)")
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "code_usage_heatmap.png"), dpi=200,
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved code_usage_heatmap.png")


# ===========================================================================
# 8. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VQGNN simulation sanity check"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-fish", type=int, default=100,
                        help="Number of fish per replicate lake")
    parser.add_argument("--n-replicates", type=int, default=5,
                        help="Number of replicate lakes per network")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--commit-alpha", type=float, default=1.0,
                        help="VQ commitment loss weight")
    parser.add_argument("--codebook-size", type=int, default=64,
                        help="Number of VQ codes")
    parser.add_argument("--output-dir", default="./simulation_output",
                        help="Output directory")
    parser.add_argument("--device", default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ---- Step 1: Define networks ----
    print("=" * 60)
    print("Step 1: Defining 10 network structures")
    print("=" * 60)
    networks = define_networks(n_genes=1000, seed=args.seed)
    for net in networks:
        print(f"  {net['name']}: {net['description']}")

    # ---- Step 2: Generate expression data ----
    print(f"\n{'='*60}")
    print("Step 2: Generating expression data")
    print(f"{'='*60}")
    matrix_dict, labels_dict = build_all_matrices(
        networks, n_fish=args.n_fish, n_replicates=args.n_replicates,
        seed=args.seed
    )
    print(f"  Generated {len(matrix_dict)} matrices: "
          f"{len(matrix_dict)} keys, each ({args.n_fish} fish x 1000 genes)")

    # ---- Step 3: Build graphs ----
    print(f"\n{'='*60}")
    print("Step 3: Building co-expression graphs")
    print(f"{'='*60}")
    graphs, radii_dict = build_graphs(matrix_dict)
    n_degenerate = sum(1 for g in graphs.values()
                       if isinstance(g, dict) and g.get("degenerate"))
    print(f"  Built {len(graphs)} graphs ({n_degenerate} degenerate)")

    # ---- Step 4: Build and train VQGNN ----
    print(f"\n{'='*60}")
    print("Step 4: Training VQGNN (sequential, shuffled)")
    print(f"{'='*60}")
    from surge.vqgnn import VQGNN

    model = VQGNN(
        n_nodes=1000,
        in_channels=64,
        hidden_channels=64,
        out_channels=16,
        num_layers=3,
        dropout=0.2,
        codebook_channels=16,
        codebook_size=args.codebook_size,
        decoder_channels=64,
        n_lakes=None,  # no lake-specific conditioning
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters, "
          f"codebook_size={args.codebook_size}")

    model = train_sequential_shuffled(
        model, graphs, radii_dict, args.output_dir,
        epochs=args.epochs, lr=args.lr,
        commit_alpha=args.commit_alpha, device=device,
    )

    # ---- Step 5: Extract embeddings ----
    print(f"\n{'='*60}")
    print("Step 5: Extracting embeddings")
    print(f"{'='*60}")
    embeddings = extract_embeddings(model, graphs, device=device)
    print(f"  Extracted {len(embeddings)} embeddings "
          f"(shape: {list(embeddings.values())[0].shape})")

    # Save embeddings
    import pickle
    emb_path = os.path.join(args.output_dir, "embeddings.pkl")
    with open(emb_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"  Saved to {emb_path}")

    # ---- Step 6: Evaluate ----
    print(f"\n{'='*60}")
    print("Step 6: Evaluation")
    print(f"{'='*60}")
    metrics = evaluate(embeddings, labels_dict, networks, args.output_dir,
                       seed=args.seed)

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    sil = metrics.get("silhouette_network")
    sil_str = f"{sil:.4f}" if sil is not None else "N/A"
    perm_p = metrics.get("silhouette_permutation_p", "N/A")
    perm_p_str = f"{perm_p:.4f}" if perm_p is not None else "N/A"
    perm_r2 = metrics.get("permanova", {}).get("network_id", {}).get("r2", "N/A")
    perm_r2_str = f"{perm_r2:.4f}" if isinstance(perm_r2, float) else "N/A"
    active = metrics.get("active_codes", "N/A")
    jsd_mean = metrics.get("pairwise_jsd_mean", "N/A")
    jsd_mean_str = f"{jsd_mean:.4f}" if jsd_mean is not None else "N/A"

    print(f"  Silhouette (network labels): {sil_str}")
    print(f"  Permutation p-value:         {perm_p_str}")
    print(f"  PERMANOVA R^2 (network_id):  {perm_r2_str}")
    print(f"  Active codes:                {active}/{args.codebook_size}")
    print(f"  Mean pairwise JS divergence: {jsd_mean_str}")

    success = (
        sil is not None and sil > 0
        and (perm_p is None or perm_p < 0.05)
    )
    if success:
        print("\n  *** VQGNN successfully recovers differences between "
              "simulated networks ***")
    else:
        print("\n  *** WARNING: VQGNN did NOT clearly separate simulated "
              "networks — investigate ***")

    print(f"\nAll outputs in: {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
