"""
CoexpressionGraphBuilder: Construct gene co-expression networks from expression matrices.

For a given expression matrix M (fish × genes), the gene-gene adjacency is:
    adj = M.T @ M

This captures how strongly each pair of genes co-varies across fish in that group.
The adjacency is then decomposed via eigendecomposition, reconstructed at multiple
scales (k = 4, 8, 32, 64, 127 components), binarized at mean threshold, and
converted to sparse edge-index format for GNN training.

Node uncertainty radii are computed as the consistency of each gene's connection
probability across reconstruction scales — genes that flip between connected and
disconnected at different scales get high radii (high uncertainty), which is used
as Gaussian noise injection during VQGNN training.
"""

import os
import re

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data
from tqdm import tqdm


class CoexpressionGraphBuilder:
    """
    Build gene co-expression networks from expression matrices.

    Parameters
    ----------
    n_eigencomponents : int (default 128)
        Number of top eigencomponents to compute.
    reconstruction_levels : list[int]
        Component counts at which to reconstruct and binarize the adjacency.
        Default: [4, 8, 32, 64, 127]
    device : str or torch.device
        Device for the resulting tensors.

    Output per graph
    ----------------
    sparse_graphs : list[torch_geometric.data.Data]
        One graph per reconstruction level, with edge_index in sparse COO format.
    radii : np.ndarray (n_genes,)
        Node uncertainty scores in [0, 1], normalized across genes.
        Higher = more inconsistent across scales = more noise injected during training.
    """

    def __init__(self, n_eigencomponents=128,
                 reconstruction_levels=None,
                 device='cpu'):
        self.n_eigencomponents = n_eigencomponents
        self.reconstruction_levels = reconstruction_levels or [4, 8, 32, 64, 127]
        self.device = device

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def build(self, expression_matrix):
        """
        Full pipeline: expression matrix → sparse graphs + radii.

        Parameters
        ----------
        expression_matrix : np.ndarray (n_fish × n_genes)
            Row-normalized expression matrix.

        Returns
        -------
        sparse_graphs : list[Data] or None
            Binarized sparse graphs at each reconstruction level.
            None if all reconstruction levels are empty (no edges).
        radii : np.ndarray (n_genes,)
            Uncertainty radii for each gene.
        """
        print ("Computing adjacency...")
        adj = self._compute_adjacency(expression_matrix)
        print ("Performing eigendecomposition...")
        eigvals, eigvecs = self._eigendecompose(adj)
        del adj  # free ~6.3 GB; no longer needed after eigendecomposition
        print ("Reconstructing graphs at multiple scales...")
        sparse_graphs, radii = self._multi_scale_reconstruct(eigvals, eigvecs)
        if not sparse_graphs:
            return None, np.ones(expression_matrix.shape[1] if expression_matrix.ndim == 2 else 0)
        # Skip degenerate graphs: all-empty or all-dense (> 99.9% edges)
        n_edges = sparse_graphs[-1].edge_index.shape[1]
        n_genes = expression_matrix.shape[1]
        max_edges = n_genes * (n_genes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0.0
        if density == 0.0:
            print("  All reconstruction levels empty — skipping this key.")
            return None, np.ones(n_genes)
        if density > 0.999:
            print(f"  Graph fully dense ({density:.4%}) — skipping this key.")
            return None, np.ones(n_genes)
        print(f"Built {len(sparse_graphs)} graphs with {len(radii)} node radii.")
        return sparse_graphs, radii

    # ------------------------------------------------------------------
    # Fast adjacency-only graph (no eigendecomposition, for inference)
    # ------------------------------------------------------------------

    def build_fast(self, expression_matrix):
        """
        Build a single sparse graph from the raw adjacency — no eigendecomposition,
        no multi-scale reconstruction. Fast path for inference-only use cases
        like permutation null models where we just need a graph to forward-pass.

        Returns a single Data, or None if degenerate.
        """
        adj = self._compute_adjacency(expression_matrix)
        n_genes = adj.shape[0]
        threshold = float(np.mean(adj))
        binary_adj = (adj >= threshold).astype(np.float32)
        np.fill_diagonal(binary_adj, 0.0)

        max_edges = n_genes * (n_genes - 1)
        n_edges = int(np.sum(binary_adj > 0))
        density = n_edges / max_edges if max_edges > 0 else 0.0
        if density == 0.0 or density > 0.999:
            return None

        rows, cols = np.where(binary_adj > 0)
        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long,
                                  device=self.device)
        return Data(edge_index=edge_index, num_nodes=n_genes)

    def build_spectral(self, expression_matrix, k=64, target_density=None):
        """
        Build a single spectrally-reconstructed graph at k eigencomponents.

        Unlike ``build_fast`` which thresholds the raw adjacency directly
        (producing dense ~65% graphs), this method performs eigendecomposition
        and reconstructs at a single k-component level — matching the
        reconstruction methodology used for the real (non-permuted) graphs.

        Parameters
        ----------
        expression_matrix : np.ndarray (n_fish, n_genes)
        k : int
            Number of eigencomponents for reconstruction.
        target_density : float or None
            If provided, binarize at a percentile threshold that yields this
            exact edge density rather than thresholding at the mean.  Used to
            density-match permuted graphs to their real counterparts so the
            null distribution is not confounded by density differences.

        Returns a single Data, or None if degenerate.
        """
        adj = self._compute_adjacency(expression_matrix)
        n_genes = adj.shape[0]
        k_actual = min(k, self.n_eigencomponents, n_genes - 2)
        if k_actual <= 0:
            return None

        eigvals, eigvecs = self._eigendecompose(adj)
        del adj  # free ~6.3 GB

        V_k = eigvecs[:, :k_actual]
        S_k = np.diag(eigvals[:k_actual])
        recon = V_k @ S_k @ V_k.T

        # Min-max scale in-place
        if np.iscomplexobj(recon):
            recon = recon.real
        r_min, r_max = recon.min(), recon.max()
        if r_max > r_min:
            recon -= r_min
            recon /= (r_max - r_min)

        # Binarize: density-matched percentile or default mean threshold
        if target_density is not None:
            # Use exact quantile — histogram bins are too coarse when the
            # permuted reconstruction has a highly skewed value distribution
            # (most values near 0 after shuffling destroys co-expression).
            threshold = float(np.quantile(recon, 1.0 - target_density))
        else:
            threshold = float(np.mean(recon))

        binary_adj = recon > threshold
        del recon
        np.fill_diagonal(binary_adj, False)

        max_edges = n_genes * (n_genes - 1)
        n_edges = int(np.sum(binary_adj))
        density = n_edges / max_edges if max_edges > 0 else 0.0
        if density == 0.0 or density > 0.999:
            print(f"  build_spectral k={k_actual}: density={density:.4%} — degenerate, skipping")
            return None

        rows, cols = np.where(binary_adj)
        tag = " (density-matched)" if target_density is not None else ""
        print(f"  build_spectral k={k_actual}: {n_edges:,} edges ({density:.2%}){tag}")
        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long,
                                  device=self.device)
        return Data(edge_index=edge_index, num_nodes=n_genes)

    # ------------------------------------------------------------------
    # Step 1: Gene co-expression adjacency
    # ------------------------------------------------------------------

    def _compute_adjacency(self, M):
        """
        Compute gene-gene co-expression adjacency: adj = M.T @ M.

        The (i,j) entry is the dot product of gene i's and gene j's expression
        profiles across all fish in the group. Min-max scaled to [0, 1].
        """
        M = np.asarray(M, dtype=np.float32)
        adj = M.T @ M
        # Min-max normalization (in-place to avoid temporaries)
        adj_min, adj_max = adj.min(), adj.max()
        if adj_max > adj_min:
            adj -= adj_min
            adj /= (adj_max - adj_min)
        return adj

    # ------------------------------------------------------------------
    # Step 2: Eigendecomposition
    # ------------------------------------------------------------------

    def _eigendecompose(self, adj):
        """
        Compute top-k eigenvalues and eigenvectors of the adjacency matrix.

        Since adj = M.T @ M, it is symmetric positive semi-definite, so all
        eigenvalues are real and non-negative. We take the top k by magnitude.
        """
        n = adj.shape[0]
        k = min(self.n_eigencomponents, n - 2)
        if k <= 0:
            return (np.array([], dtype=np.float32),
                    np.zeros((n, 0), dtype=np.float32))

        # Degenerate adjacency (all zeros) can trigger ARPACK failure.
        if not np.any(adj):
            print ("  Adjacency all zeros — skipping eigendecomposition and returning empty eigenvectors.")
            return np.zeros(k, dtype=np.float32), np.eye(n, k, dtype=np.float32)

        # Provide an explicit non-zero start vector to stabilize ARPACK.
        v0 = np.full(n, 1.0 / np.sqrt(n), dtype=np.float32)
        try:
            vals, vecs = eigsh(adj, k=k, which='LM')
        except Exception:
            # Retry with a tiny diagonal jitter for near-degenerate matrices.
            vals, vecs = eigsh(adj + 1e-8 * np.eye(n, dtype=np.float32),
                               k=k, which='LM', v0=v0)
        # eigsh guarantees real eigenvalues for symmetric PSD matrices.
        # Cast to float32 so the downstream n×n reconstruction stays float32
        # (halves memory) regardless of the dtype eigsh returns.
        vals = np.real(vals).astype(np.float32, copy=False)
        vecs = np.real(vecs).astype(np.float32, copy=False)
        # Sort by descending eigenvalue
        order = np.argsort(vals)[::-1]
        return vals[order], vecs[:, order]

    # ------------------------------------------------------------------
    # Step 3: Multi-scale reconstruction & binarization
    # ------------------------------------------------------------------

    def _multi_scale_reconstruct(self, eigvals, eigvecs):
        """
        Reconstruct adjacency at multiple scales, binarize, and compute
        node uncertainty radii.

        For each k in reconstruction_levels:
            recon = V_k @ diag(S_k) @ V_k.T   (truncated spectral reconstruction)
        where V_k = first k eigenvectors, S_k = first k eigenvalues.

        Each reconstruction is min-max scaled to [0, 1] and binarized at its
        own per-level mean threshold.

        Node uncertainty = 1 - avg(|2x-1|) across scales.
        This measures how often a gene's connections are "ambiguous" (near 0.5
        probability) — genes with consistently extreme values (near 0 or 1) get
        low uncertainty; genes that flip get high uncertainty.

        Memory-conscious: processes one reconstruction level at a time, fusing
        binarization and radii accumulation to avoid storing all dense
        reconstructions simultaneously.
        """
        n_genes = eigvecs.shape[0]
        cus_eig = np.zeros(n_genes)
        sparse_graphs = []
        edge_counts = []
        max_edges = n_genes * (n_genes - 1)
        n_levels = 0

        for k in tqdm(self.reconstruction_levels, desc="Reconstructing graphs"):
            if k > eigvecs.shape[1]:
                continue

            # ---- Build reconstruction ----
            V_k = eigvecs[:, :k]
            S_k = np.diag(eigvals[:k])
            recon = V_k @ S_k @ V_k.T

            # ---- Min-max scale in-place ----
            if np.iscomplexobj(recon):
                recon = recon.real
            r_min = recon.min()
            r_max = recon.max()
            if r_max > r_min:
                recon -= r_min
                recon /= (r_max - r_min)

            # ---- Binarize (non-destructive: creates boolean array) ----
            threshold = float(np.mean(recon))
            binary_adj = recon > threshold
            np.fill_diagonal(binary_adj, False)
            rows, cols = np.where(binary_adj)
            n_edges = len(rows)
            edge_counts.append(n_edges)

            edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long,
                                      device=self.device)
            # Store target as bool (791 MB vs 3.1 GB for float32).
            # reconstruction_loss handles bool correctly (float32 - bool → float32).
            target_adj = torch.tensor(binary_adj)
            sparse_graphs.append(Data(edge_index=edge_index,
                                      target_adj=target_adj,
                                      num_nodes=n_genes))

            # ---- Radii contribution (in-place on recon, destroys it) ----
            # Compute per_gene = 1.0 - mean(|2*recon - 1.0|, axis=1)
            np.multiply(recon, 2.0, out=recon)
            np.subtract(recon, 1.0, out=recon)
            np.abs(recon, out=recon)
            per_gene = 1.0 - np.mean(recon, axis=1)
            cus_eig += per_gene
            n_levels += 1
            # recon freed on next iteration; binary_adj freed on next iteration

        if n_levels == 0:
            return [], np.ones(n_genes)

        # Normalize radii
        cus_eig /= n_levels
        cus_min, cus_max = cus_eig.min(), cus_eig.max()
        if cus_max > cus_min:
            radii = (cus_eig - cus_min) / (cus_max - cus_min)
        else:
            radii = np.ones(n_genes)

        # Print per-level densities
        density_strs = []
        for k, n_e in zip(self.reconstruction_levels[:len(edge_counts)], edge_counts):
            density_strs.append(f"k={k}: {n_e:,} edges ({n_e/max_edges:.4%})")
        print(f"  Densities — {' | '.join(density_strs)}")

        return sparse_graphs, radii

    # ------------------------------------------------------------------
    # Batch build
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_key(key):
        """Convert arbitrary dict keys into filesystem-safe file stems."""
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(key)).strip('._')
        return safe or 'graph'

    @staticmethod
    def _infer_stratification(key):
        """Infer stratification family from canonical key formatting."""
        key_str = str(key)
        if re.search(r'\)-[fFmM]$', key_str):
            return 'sex_year_lake'
        if re.search(r'\)-[01]$', key_str):
            return 'infection_year_lake'
        if re.search(r'\(\d{4}\)$', key_str):
            return 'year_lake'
        return 'lake'

    @staticmethod
    def is_degenerate(graph_entry):
        """Check if a graph entry (in-memory or bundle path) is a degenerate sentinel."""
        if isinstance(graph_entry, dict):
            return graph_entry.get('degenerate', False)
        if isinstance(graph_entry, (str, os.PathLike)):
            try:
                payload = torch.load(os.fspath(graph_entry), map_location='cpu',
                                     weights_only=False)
                return bool(payload.get('degenerate', False)) if isinstance(payload, dict) else False
            except Exception:
                return False
        return False

    @staticmethod
    def _coerce_radii_array(radii_obj, n_genes=None):
        """Convert stored radii payloads to float32 numpy arrays."""
        if radii_obj is None:
            if n_genes is None:
                raise ValueError("Missing radii payload and unknown n_genes.")
            return np.ones(n_genes, dtype=np.float32)

        if isinstance(radii_obj, torch.Tensor):
            arr = radii_obj.detach().cpu().numpy()
        else:
            arr = np.asarray(radii_obj)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _graph_stats_row(key, stratification, M, graph_list, reused=False):
        """Extract one row of stats for a single key's graphs."""
        n_fish = M.shape[0]
        n_genes = M.shape[1]
        edge_counts = []
        for g in graph_list:
            ei = getattr(g, 'edge_index', None)
            if ei is None and isinstance(g, dict):
                ei = g.get('edge_index')
            edge_counts.append(int(ei.shape[1]) if ei is not None else 0)
        return {
            'key': str(key),
            'stratification': stratification,
            'n_fish': n_fish,
            'n_genes': n_genes,
            'n_levels': len(graph_list),
            'edges_per_level': edge_counts,
            'reused': reused,
        }

    @staticmethod
    def _write_graph_stats(output_dir, stats_rows):
        """Write per-graph stats CSV to output_dir/graph_stats.csv."""
        import csv
        csv_path = os.path.join(output_dir, 'graph_stats.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'key', 'stratification', 'n_fish', 'n_genes', 'n_levels',
                'edges_per_level', 'reused',
            ])
            for row in stats_rows:
                writer.writerow([
                    row['key'], row['stratification'], row['n_fish'],
                    row['n_genes'], row['n_levels'],
                    '|'.join(str(v) for v in row['edges_per_level']),
                    int(row['reused']),
                ])
        print(f"Graph stats saved to {csv_path}")

    @staticmethod
    def _config_hash(n_eigencomponents, reconstruction_levels):
        """Short hash of graph construction parameters for cache validation."""
        import hashlib
        payload = f"{n_eigencomponents}|{'_'.join(map(str, reconstruction_levels))}"
        return hashlib.md5(payload.encode()).hexdigest()[:8]

    @staticmethod
    def _find_bundle(output_dir, stratification, safe_key):
        """Find an existing bundle file for a key, checking new and old formats."""
        import glob as _glob
        new_path = os.path.join(output_dir, f"{stratification}_{safe_key}.pt")
        if os.path.exists(new_path):
            return new_path
        # Backward-compatible: old format used {idx:04d}_ prefix
        pattern = os.path.join(output_dir, f"*_{stratification}_{safe_key}.pt")
        matches = _glob.glob(pattern)
        if matches:
            return matches[0]
        return new_path  # doesn't exist, but return canonical path for saving

    @staticmethod
    def _bundle_radii_path(output_dir, stratification, safe_key):
        """Return the canonical radii sidecar path."""
        return os.path.join(output_dir, f"{stratification}_{safe_key}.radii.npy")

    @staticmethod
    def _validate_bundle(payload, key, expected_config_hash):
        """Return True if a loaded bundle is usable. Prints reason on failure."""
        if not isinstance(payload, dict):
            print("  Bundle unusable: not a dict")
            return False
        stored_key = payload.get('key', key)
        if str(stored_key) != str(key):
            print(f"  Bundle unusable: key mismatch ({stored_key} != {key})")
            return False
        stored_hash = payload.get('config_hash', '')
        if stored_hash and stored_hash != expected_config_hash:
            print(f"  Bundle unusable: config hash mismatch "
                  f"({stored_hash} != {expected_config_hash})")
            return False
        if payload.get('degenerate'):
            # Degenerate bundles are always valid — no graphs to mismatch
            return True
        if 'graphs' not in payload:
            print("  Bundle unusable: missing 'graphs' key")
            return False
        return True

    def build_all(self, matrix_dict, output_dir=None, keep_in_memory=True,
                  reuse_existing=True):
        """
        Build graphs for all expression matrices in a dictionary.

        Parameters
        ----------
        matrix_dict : dict {key: np.ndarray}
            Maps descriptive keys to expression matrices.
        output_dir : str or None, optional
            If provided, each key's graphs+radii are serialized to a separate
            .pt bundle in this directory.
        keep_in_memory : bool, optional
            If False and output_dir is provided, the returned `graphs` dict
            stores bundle file paths instead of Data objects.
        reuse_existing : bool, optional
            If True and output bundles already exist, skip recomputing those
            keys and reuse the existing bundle/radii payloads.

        Returns
        -------
        graphs : dict {key: list[Data] or str}
            Sparse graphs for each key, or bundle paths when
            output_dir is set and keep_in_memory=False.
            Degenerate keys are stored as a dict with ``degenerate=True``.
        radii_dict : dict {key: np.ndarray}
            Uncertainty radii for each key.
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        config_hash = self._config_hash(
            self.n_eigencomponents, self.reconstruction_levels,
        )

        graphs = {}
        radii_dict = {}
        reused_count = 0
        built_count = 0
        degenerate_count = 0
        degenerate_keys_set = set()  # populated during loop; avoids a
        # torching-load-every-.pt scan at the end to build the manifest.
        stats_rows = []

        for key, M in tqdm(matrix_dict.items(),
                                            desc="Building graphs"):
            bundle_path = None
            radii_sidecar = None
            stratification = self._infer_stratification(key)
            safe = self._safe_key(key)
            if output_dir is not None:
                bundle_path = self._find_bundle(output_dir, stratification, safe)
                radii_sidecar = self._bundle_radii_path(output_dir, stratification, safe)

            if output_dir is not None and reuse_existing and os.path.exists(bundle_path):
                try:
                    payload = torch.load(bundle_path, map_location='cpu',
                                         weights_only=False)
                    if self._validate_bundle(payload, key, config_hash):
                        key_radii = None
                        if radii_sidecar is not None and os.path.exists(radii_sidecar):
                            key_radii = self._coerce_radii_array(np.load(radii_sidecar))
                        else:
                            key_radii = self._coerce_radii_array(
                                payload.get('radii'), n_genes=M.shape[1],
                            )
                            if radii_sidecar is not None:
                                np.save(radii_sidecar, key_radii)

                        radii_dict[key] = key_radii
                        if keep_in_memory:
                            if payload.get('degenerate'):
                                graphs[key] = {'degenerate': True,
                                               'key': key,
                                               'stratification': stratification,
                                               'radii': key_radii,
                                               'n_genes': M.shape[1]}
                            else:
                                graphs[key] = payload['graphs']
                        else:
                            graphs[key] = bundle_path
                        reused_count += 1
                        if payload.get('degenerate'):
                            degenerate_keys_set.add(str(key))
                            stats_rows.append({
                                'key': str(key),
                                'stratification': stratification,
                                'n_fish': M.shape[0],
                                'n_genes': M.shape[1],
                                'n_levels': 0,
                                'edges_per_level': [],
                                'reused': True,
                                'degenerate': True,
                            })
                        else:
                            glist = payload['graphs']
                            if not isinstance(glist, (list, tuple)):
                                glist = [glist]
                            stats_rows.append(self._graph_stats_row(
                                key, stratification, M, glist, reused=True))
                        continue
                except Exception as exc:
                    print(f"Recomputing {key} (existing bundle unusable: {exc})")

            key_graphs, key_radii = self.build(M)
            radii_dict[key] = key_radii

            if key_graphs is None:
                # Degenerate graph — save a sentinel bundle so the key
                # persists for clustering and downstream analysis.
                if output_dir is not None and bundle_path is not None:
                    torch.save(
                        {
                            'degenerate': True,
                            'key': key,
                            'stratification': stratification,
                            'radii': torch.tensor(key_radii, dtype=torch.float32),
                            'n_genes': M.shape[1],
                            'config_hash': config_hash,
                        },
                        bundle_path,
                    )
                    if radii_sidecar is not None:
                        np.save(radii_sidecar, np.asarray(key_radii, dtype=np.float32))
                    graphs[key] = (
                        {'degenerate': True, 'key': key,
                         'stratification': stratification,
                         'radii': key_radii, 'n_genes': M.shape[1]}
                        if keep_in_memory else bundle_path
                    )
                degenerate_keys_set.add(str(key))
                degenerate_count += 1
                stats_rows.append({
                    'key': str(key),
                    'stratification': stratification,
                    'n_fish': M.shape[0],
                    'n_genes': M.shape[1],
                    'n_levels': 0,
                    'edges_per_level': [],
                    'reused': False,
                    'degenerate': True,
                })
                continue

            if output_dir is not None and bundle_path is not None:
                torch.save(
                    {
                        'key': key,
                        'stratification': stratification,
                        'graphs': key_graphs,
                        'radii': torch.tensor(key_radii, dtype=torch.float32),
                        'config_hash': config_hash,
                    },
                    bundle_path,
                )
                if radii_sidecar is not None:
                    np.save(radii_sidecar, np.asarray(key_radii, dtype=np.float32))
                graphs[key] = key_graphs if keep_in_memory else bundle_path
            else:
                graphs[key] = key_graphs
            built_count += 1
            stats_rows.append(self._graph_stats_row(key, stratification, M, key_graphs, reused=False))

        if output_dir is not None:
            self._write_graph_stats(output_dir, stats_rows)
            # Save degenerate manifest so downstream validation is instant.
            # Populated during the loop so we don't torch.load every .pt file
            # (2-12 GB each) just to read a boolean flag.
            degenerate_keys = sorted(str(k) for k in degenerate_keys_set)
            import json as _json
            manifest_path = os.path.join(output_dir, 'degenerate_manifest.json')
            with open(manifest_path, 'w') as f:
                _json.dump(sorted(str(k) for k in degenerate_keys), f)
            if reuse_existing:
                print(f"Graph build summary: reused={reused_count}, "
                      f"built={built_count}, degenerate={degenerate_count}, "
                      f"total={len(matrix_dict)}")

        return graphs, radii_dict
