"""
LakeEmbedder: Train VQGNN on lake gene networks and generate VQ-code histogram
embeddings for downstream comparison.

Each lake (or lake-year / lake-year-sex / lake-year-infection stratification)
gets a normalized histogram over the VQ codebook as its "embedding". These
histograms encode the discrete structure of the gene co-expression network
and can be compared via PCA, Wasserstein distance, silhouette score, etc.

The embedder manages the full lifecycle:
    1. Train VQGNN on graphs (or load pretrained weights).
    2. Run forward pass to get VQ code assignments for each gene.
    3. Aggregate assignments into a histogram → lake embedding.
    4. Build gene↔VQ-code lookup tables for gene-level analysis.
    5. Generate permuted null models via sample shuffling.
"""

import os
import numpy as np
import torch
import pickle
from collections import defaultdict
from tqdm import tqdm

class LakeEmbedder:
    """
    Trains VQGNN and generates lake embeddings from VQ code histograms.

    Parameters
    ----------
    model : VQGNN
        The VQGNN model instance.
    device : str or torch.device, optional
        Device for training/inference. Defaults to CUDA if available.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Graph loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_graph_entry(graph_entry):
        """
        Resolve either an in-memory graph list or a disk bundle path.

        Bundle format is expected to be a torch-saved dict with keys:
        {'graphs': list[Data], 'radii': tensor/array (optional)}.
        """
        bundled_radii = None
        if isinstance(graph_entry, (str, os.PathLike)):
            payload = torch.load(os.fspath(graph_entry), map_location='cpu',
                                 weights_only=False)
            if isinstance(payload, dict) and 'graphs' in payload:
                graphs = payload['graphs']
                bundled_radii = payload.get('radii')
            else:
                graphs = payload
        else:
            graphs = graph_entry

        if isinstance(bundled_radii, torch.Tensor):
            bundled_radii = bundled_radii.detach().cpu().numpy()
        elif bundled_radii is not None:
            bundled_radii = np.asarray(bundled_radii)

        return graphs, bundled_radii

    def _get_primary_graph(self, graph_entry):
        """Return the first reconstruction graph and optional bundled radii."""
        graphs, bundled_radii = self._load_graph_entry(graph_entry)

        if isinstance(graphs, (list, tuple)):
            if not graphs:
                raise ValueError("Encountered an empty graph list.")
            graph = graphs[0]
        else:
            graph = graphs

        if isinstance(graph, dict) and graph.get('degenerate'):
            raise ValueError("Cannot get primary graph from degenerate entry — "
                             "check CoexpressionGraphBuilder.is_degenerate() first")
        if not hasattr(graph, 'edge_index'):
            raise TypeError("Graph entry must provide an edge_index tensor.")

        return graph, bundled_radii

    @staticmethod
    def _resolve_radii_for_key(key, radii_dict, bundled_radii, n_nodes):
        """Resolve radii from explicit dict, bundle payload, or fallback ones."""
        radii_value = None

        if radii_dict is not None and key in radii_dict:
            radii_value = radii_dict[key]
        elif bundled_radii is not None:
            radii_value = bundled_radii

        if radii_value is None:
            return np.ones(n_nodes, dtype=np.float32)

        if isinstance(radii_value, (str, os.PathLike)):
            arr = np.load(os.fspath(radii_value))
        elif isinstance(radii_value, torch.Tensor):
            arr = radii_value.detach().cpu().numpy()
        else:
            arr = np.asarray(radii_value)

        return arr.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, graphs_dict, radii_dict, save_path=None,
              epochs=5, lr=1e-4, commit_alpha=0.25, noise_scale=5.0,
              recon_batch_size=256, skip_oom=True, preloaded_seq=None):
        """
        Train VQGNN sequentially on each graph in ``graphs_dict``.

        Resumes from checkpoint if ``save_path`` is provided and a prior
        run was interrupted (e.g., Modal preemption).

        Parameters
        ----------
        graphs_dict : dict {key: list[Data] or str}
            Sparse graphs per key, or bundle paths to torch-saved graphs.
        radii_dict : dict {key: np.ndarray}
            Uncertainty radii for each lake. Values may be arrays or .npy paths.
        save_path : str, optional
            Path to save model weights and training progress.
        epochs : int
            Epochs per graph.
        lr : float
            Learning rate.
        commit_alpha : float
            Weight for VQ commitment loss.
        noise_scale : float
            Multiplier for radii (data augmentation noise level).
            Use 0 to disable noise injection.
        recon_batch_size : int
            Row batch size for blockwise reconstruction loss.
            Lower values reduce peak memory.
        skip_oom : bool
            If True, CUDA OOM on an individual graph will skip that graph
            and continue training on remaining keys.
        preloaded_seq : list or None
            Pre-loaded graph data: list of (key, edge_index, target_adj,
            scaled_radii, recon_weight).  If provided, ``graphs_dict`` and
            ``radii_dict`` are ignored for loading.
        """
        keys = sorted(graphs_dict.keys())
        device = next(self.model.parameters()).device
        skipped_oom = []

        # Resume model weights if interrupted mid-training
        if save_path is not None and os.path.exists(save_path):
            print(f"[train] Resuming model weights from {save_path}")
            self.model.load_state_dict(
                torch.load(save_path, map_location=device, weights_only=True))

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                     weight_decay=1e-4)

        if preloaded_seq is not None:
            graph_data = preloaded_seq
            print(f"[train] Using {len(graph_data)} pre-loaded graphs")
        else:
            # Pre-load all valid graphs once.  Each bundle holds one graph
            # per reconstruction level (densities ramping 1% → 2% → …); we
            # train on EVERY level and scale the reconstruction loss by
            # 1/(i+1) for level i so the denser levels — whose MSE is
            # naturally larger — do not dominate.
            graph_data = []  # (key, edge_index, target_adj, scaled_radii, recon_weight)
            oom_set = set(skipped_oom)
            for key in tqdm(keys, desc="  Loading graphs", unit="graph"):
                if key in oom_set:
                    continue
                payload = self._load_graph_entry(graphs_dict[key])
                graphs, bundled_radii = payload
                if not isinstance(graphs, (list, tuple)):
                    graphs = [graphs]
                if not graphs:
                    continue
                if (isinstance(graphs[0], dict)
                        and graphs[0].get('degenerate')):
                    continue
                radii_arr = self._resolve_radii_for_key(
                    key=key, radii_dict=radii_dict,
                    bundled_radii=bundled_radii,
                    n_nodes=graphs[0].num_nodes,
                )
                scaled_radii = radii_arr * float(noise_scale)
                for i, g in enumerate(graphs):
                    if not hasattr(g, 'edge_index'):
                        continue
                    graph_data.append((key, g.edge_index, g.target_adj,
                                       scaled_radii, 1.0 / (i + 1)))

        n_graphs = len(graph_data)
        print(f"[train] {n_graphs} valid graphs loaded, {epochs} epochs")

        oom_skip_idx = set()  # blacklisted after first OOM, never retried
        # Best-epoch tracking: checkpoint the epoch with the lowest
        # mean(edge_loss) + 0.1 * mean(commit_loss) — reconstruction quality
        # with a light VQ/codebook signal, rather than the final epoch.
        best = {'state': None, 'crit': float('inf'),
                'epoch': -1}
        for ep in range(epochs):
            active = [i for i in range(n_graphs) if i not in oom_skip_idx]
            order = np.random.permutation(active)
            pbar = tqdm(order, desc=f"Epoch {ep+1}/{epochs}", unit="graph",
                        leave=False)
            ep_losses = []
            ep_edge_losses = []
            ep_commit_losses = []
            for g_idx in pbar:
                key, edge_index, target_adj, radii_arr, recon_weight = \
                    graph_data[g_idx]
                r = (torch.tensor(radii_arr, dtype=torch.float32, device=device)
                     if noise_scale != 0 else None)
                try:
                    _, decoded, _, _, commit_loss = self.model.forward(
                        edge_index.to(device),
                        radii=r,
                    )
                    edge_loss = self.model.reconstruction_loss(
                        decoded,
                        target_adj,
                        batch_size=recon_batch_size,
                    )
                    # Scale the reconstruction term by 1/(i+1) for level i so
                    # denser reconstruction levels don't dominate the loss.
                    loss = recon_weight * edge_loss + commit_alpha * commit_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    ep_losses.append(loss.item())
                    ep_edge_losses.append(float(edge_loss.item()))
                    ep_commit_losses.append(float(commit_loss.item()))
                    pbar.set_postfix(loss=f"{loss.item():.4f}",
                                     avg=f"{np.mean(ep_losses):.4f}")

                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or \
                        ('out of memory' in str(exc).lower())
                    if not is_oom or not skip_oom:
                        raise
                    skipped_oom.append(str(key))
                    oom_skip_idx.add(g_idx)
                    tqdm.write(f"\n[OOM] Skipping graph {key} "
                               f"(blacklisted for remaining epochs)")
                    optimizer.zero_grad(set_to_none=True)
                    # Only clear the cache on the OOM-recovery path, where we
                    # genuinely need to release the failed allocation. On the
                    # normal path the caching allocator reuses blocks without
                    # a forced sync.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Track the best epoch by edge + 0.1*commit (means over the
            # graphs actually trained this epoch).
            if ep_edge_losses:
                epoch_crit = (float(np.mean(ep_edge_losses))
                              + 0.1 * float(np.mean(ep_commit_losses)))
                if epoch_crit < best['crit']:
                    best = {
                        'state': {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()},
                        'crit': epoch_crit,
                        'epoch': ep + 1,
                    }
            best_str = (f" | best ep {best['epoch']} {best['crit']:.4f}"
                        if best['state'] is not None else "")
            print(f"Epoch {ep+1}/{epochs} | "
                  f"avg loss: {np.mean(ep_losses):.4f} | "
                  f"min: {np.min(ep_losses):.4f} | "
                  f"max: {np.max(ep_losses):.4f}{best_str}")

            if save_path is not None:
                torch.save(self.model.state_dict(), save_path)

        if skipped_oom:
            print(f"Training summary: {n_graphs} graphs trained, "
                  f"skipped_oom={len(skipped_oom)}")
            preview = ', '.join(skipped_oom[:5])
            if preview:
                print(f"OOM-skipped keys (first up to 5): {preview}")

        # Restore the best epoch and save it as the checkpoint, so downstream
        # steps reflect the best epoch.
        if best['state'] is not None:
            self.model.load_state_dict(best['state'])
            if save_path is not None:
                torch.save(self.model.state_dict(), save_path)
            print(f"[train] Best epoch {best['epoch']} "
                  f"(criterion={best['crit']:.4f}) saved to {save_path}")
        elif save_path is not None:
            torch.save(self.model.state_dict(), save_path)

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, graph, lake_idx=None):
        """
        Generate VQ code histogram embedding for a single graph.

        Parameters
        ----------
        graph : Data
            Sparse graph (uses first reconstruction level if list).
        lake_idx : int or None
            Lake identity for conditioning.  REQUIRED for a jointly-trained
            model (which shifts gene representations by a learned per-lake
            vector before VQ discretization); passing None silently drops
            that conditioning.  Only meaningful when the model has lake_emb.

        Returns
        -------
        hist : np.ndarray (codebook_size,)
            Normalized histogram over VQ codes.
        """
        self.model.eval()
        edge_index = graph.edge_index.to(self.device)
        return self.model.get_codebook_histogram(
            edge_index, lake_idx=lake_idx).cpu().numpy()

    def _resolve_lake_idx(self, key, lake_name_to_id):
        """Resolve a lake index for a key, raising loudly if a joint model is
        used without the mapping it needs (never silently drop conditioning)."""
        if getattr(self.model, 'lake_emb', None) is None:
            return None
        if not lake_name_to_id:
            raise ValueError(
                "Joint-trained model (lake_emb present) requires a "
                "lake_name_to_id mapping at embedding/mapping time; got None. "
                "Without it the per-lake conditioning learned during training "
                "would be silently dropped.")
        lake = str(key).split(' (')[0]
        idx = lake_name_to_id.get(lake)
        if idx is None:
            raise KeyError(f"Lake '{lake}' (from key '{key}') missing from "
                           f"lake_name_to_id — cannot condition joint model.")
        return idx

    def embed_all(self, graphs_dict, lake_name_to_id=None):
        """
        Generate embeddings for all graphs.

        Parameters
        ----------
        graphs_dict : dict {key: list[Data] or str}
            Multi-scale graphs per key, or bundle paths.
        lake_name_to_id : dict {str: int} or None
            Lake-name → index mapping used to condition a jointly-trained
            model.  Required when the model has lake_emb (else a loud
            ValueError is raised rather than silently dropping conditioning).

        Returns
        -------
        embeddings : dict {key: np.ndarray}
            VQ code histogram for each key.
        """
        embeddings = {}
        from .graphs import CoexpressionGraphBuilder as _CGB
        codebook_size = self.model.vq.codebook_size
        for key, graph_entry in tqdm(graphs_dict.items(),
                                     desc="  Generating embeddings",
                                     unit="key"):
            if _CGB.is_degenerate(graph_entry):
                embeddings[key] = np.ones(codebook_size) / codebook_size
                continue
            graph, _ = self._get_primary_graph(graph_entry)
            lake_idx = self._resolve_lake_idx(key, lake_name_to_id)
            embeddings[key] = self.embed(graph, lake_idx=lake_idx)
        return embeddings

    # ------------------------------------------------------------------
    # Gene ↔ VQ code mappings
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_assignments(self, graph, lake_idx=None):
        """Return (gene_index, vq_code) pairs for all genes in a graph."""
        edge_index = graph.edge_index.to(self.device)
        indices = self.model.get_vq_assignments(
            edge_index, lake_idx=lake_idx).cpu().numpy()
        return indices

    def build_gene_vq_mappings(self, graphs_dict, lake_name_to_id=None):
        """
        Build bidirectional gene ↔ VQ code lookup tables.

        For each stratification key, records which genes map to each
        VQ code and which VQ codes each gene maps to.  For a jointly-trained
        model, ``lake_name_to_id`` is required so the per-lake conditioning
        learned at training time is applied here too (never silently dropped).

        Returns
        -------
        vq_to_gene : dict {key: dict {vq_code: list[gene_idx]}}
            For each key, maps VQ code → list of gene indices.
        gene_to_vq : dict {key: dict {gene_idx: list[vq_code]}}
            For each key, maps gene index → list of VQ codes.
        """
        vq_to_gene = {}
        gene_to_vq = {}

        from .graphs import CoexpressionGraphBuilder as _CGB
        for key, graph_entry in tqdm(graphs_dict.items(),
                                     desc="  Building gene-VQ mappings",
                                     unit="key"):
            if _CGB.is_degenerate(graph_entry):
                vq_to_gene[key] = {}
                gene_to_vq[key] = {}
                continue
            graph, _ = self._get_primary_graph(graph_entry)
            lake_idx = self._resolve_lake_idx(key, lake_name_to_id)
            assign = self._get_assignments(graph, lake_idx=lake_idx)

            vq_to_gene[key] = defaultdict(list)
            gene_to_vq[key] = defaultdict(list)

            for gene_idx, code in enumerate(assign):
                vq_to_gene[key][code].append(gene_idx)
                gene_to_vq[key][gene_idx].append(code)

            # Convert defaultdicts to regular dicts
            vq_to_gene[key] = dict(vq_to_gene[key])
            gene_to_vq[key] = dict(gene_to_vq[key])

        return vq_to_gene, gene_to_vq

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    # DEAD CODE (unused): save/load — models are saved/loaded directly via
    # torch.save(model.state_dict(), path) / model.load_state_dict(torch.load(...))
    # everywhere in pipeline.py and modal_pipeline.py.
    # def save(self, path):
    #     torch.save(self.model.state_dict(), path)

    # def load(self, path):
    #     self.model.load_state_dict(torch.load(path, map_location=self.device,
    #                                           weights_only=True))
    #     self.model.to(self.device)

    # DEAD CODE (unused): save_embeddings / load_embeddings — embeddings are
    # pickled directly via save_pickle() in pipeline.py.
    # @staticmethod
    # def save_embeddings(embeddings, path):
    #     with open(path, 'wb') as f:
    #         pickle.dump(embeddings, f)

    # @staticmethod
    # def load_embeddings(path):
    #     with open(path, 'rb') as f:
    #         return pickle.load(f)

    # ═══════════════════════════════════════════════════════════════════════
    # DEAD CODE (unused): generate_permuted_assignments and
    # _generate_permuted_sequential — permutation null models for gene
    # co-occurrence analysis.  Replaced by density-matched permutation in
    # pipeline.py step 8 (step_label_permutation) and scripts/.
    # Kept as reference; uncomment if needed for standalone null-model runs.
    # ═══════════════════════════════════════════════════════════════════════
    #
    # @torch.no_grad()
    # def generate_permuted_assignments(self, expression_dict, graph_builder,
    #                                   n_permutations=100, seed=42,
    #                                   spectral_k=None, verbose=True):
    #     ...  # ~120 lines — see git history
    #
    # @torch.no_grad()
    # def _generate_permuted_sequential(self, expression_dict, graph_builder,
    #                                   n_permutations=100, seed=42,
    #                                   spectral_k=None, verbose=True):
    #     ...  # ~115 lines — see git history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
