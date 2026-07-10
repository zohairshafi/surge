"""
VQGNN: Vector-Quantized Graph Neural Network for gene co-expression networks.

This model learns discrete latent representations of genes by:
1. Embedding each gene into a continuous latent space (learned Embedding).
2. Passing embeddings through GraphSAGE convolution layers.
3. Discretizing node representations via Vector Quantization (VQ).
4. Decoding quantized vectors back to a higher-dimensional space.
5. Reconstructing the adjacency matrix via inner products of decoded vectors.

The VQ code assignments serve as a discrete "fingerprint" for each lake's
gene network — lakes with similar code distributions have similar network structure.

Architecture
------------
Embedding(n_genes, in_dim) → SAGEConv × num_layers → VQ(codebook_size, dim) → Linear(decoder)

Uses `vector-quantize-pytorch` (`pip install vector-quantize-pytorch`) for the
VQ layer, replacing the previously vendored vqgraph submodule.

Key fixes from original:
- blockwise_loss: removed per-block min-max scaling that caused inconsistent
  normalization across blocks. The adjacency target is already in [0,1],
  so the inner-product reconstruction is sigmoid-squashed instead.
- __init__: removed the `self = self.to(self.device)` anti-pattern.
  The model is moved to device externally via `model.to(device)`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize


class VQGNN(nn.Module):
    """
    Vector-Quantized Graph Neural Network.

    Parameters
    ----------
    n_nodes : int
        Number of genes (nodes in the co-expression graph).
    in_channels : int
        Dimension of the learned gene embedding.
    hidden_channels : int
        Hidden dimension for SAGEConv layers.
    out_channels : int
        Output dimension after SAGEConv (must equal codebook_channels —
        the SAGEConv output is the direct input to the VQ layer).
    num_layers : int
        Number of SAGEConv layers (≥ 2).
    dropout : float
        Dropout probability applied after each SAGEConv (except last).
    codebook_channels : int
        Dimension of each VQ code vector.
    codebook_size : int
        Number of discrete codes in the VQ codebook.
    decoder_channels : int
        Output dimension of the decoder linear layer.
        Reconstructed adjacency is decoded @ decoded.T.
    n_lakes : int, optional
        Number of distinct lakes for lake-identity conditioning.
        When provided, a learned lake embedding is added to node
        representations before VQ discretization (see ``forward``).
    lake_embed_dim : int, optional
        Dimension of the learned lake embedding (default 32).
        Only used when ``n_lakes`` is provided.
    """

    def __init__(self, n_nodes, in_channels, hidden_channels, out_channels,
                 num_layers, dropout, codebook_channels, codebook_size,
                 decoder_channels, n_lakes=None, lake_embed_dim=32):
        super().__init__()

        self.node_emb = nn.Embedding(n_nodes, in_channels)

        # Build SAGEConv stack
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        # Vector quantization layer (from pip package, not vqgraph submodule).
        # Anti-collapse measures: k-means init, dead-code revival, lower-dim
        # codebook (Improved VQGAN), orthogonal regularisation, faster EMA.
        self.vq = VectorQuantize(
            dim=codebook_channels,
            codebook_size=codebook_size,
            codebook_dim=8,
            use_cosine_sim=True,
            kmeans_init=True,
            kmeans_iters=20,
            threshold_ema_dead_code=2,
            orthogonal_reg_weight=10,
            orthogonal_reg_active_codes_only=True,
            decay=0.7,
        )

        self.decoder = nn.Linear(codebook_channels, decoder_channels)
        self.dropout = dropout

        # Lake-identity conditioning (optional).
        # When enabled, each lake index gets a learned embedding that is
        # projected to codebook_channels and ADDED to every gene's SAGEConv
        # output before VQ discretization.  This shifts the gene
        # representations in a lake-specific direction, so the VQ layer can
        # assign different codes to the same gene across different lakes.
        self.n_lakes = n_lakes
        if n_lakes is not None:
            self.lake_emb = nn.Embedding(n_lakes, lake_embed_dim)
            self.lake_proj = nn.Linear(lake_embed_dim, codebook_channels)
        else:
            self.lake_emb = None
            self.lake_proj = None

    # DEAD CODE (unused): reset_parameters — never called externally.
    # All conv layers are initialized in __init__ via SAGEConv defaults.
    # def reset_parameters(self):
    #     for conv in self.convs:
    #         conv.reset_parameters()

    def forward(self, edge_index, radii=None, lake_idx=None):
        """
        Forward pass.

        Parameters
        ----------
        edge_index : torch.Tensor (2, n_edges)
            Sparse COO edge index.
        radii : torch.Tensor (n_nodes,) or None
            Per-node uncertainty radii. If provided, Gaussian noise with
            std = radii is added to node representations after each conv
            (data augmentation).
        lake_idx : int or torch.Tensor (scalar), optional
            Lake index for identity conditioning.  When provided, a learned
            lake embedding (projected to codebook_channels) is added to every
            gene's representation before VQ discretization.  This lets the
            model learn lake-specific shifts in the code assignment.

        Returns
        -------
        x : torch.Tensor (n_nodes, out_channels)
            Pre-VQ node representations (before lake shift, if any).
        decoded : torch.Tensor (n_nodes, decoder_channels)
            Decoded representations. Adjacency ≈ decoded @ decoded.T
        quantized : torch.Tensor (n_nodes, codebook_channels)
            Post-VQ quantized representations.
        indices : torch.Tensor (n_nodes,)
            VQ code assignment for each gene.
        commit_loss : torch.Tensor (scalar)
            Commitment loss encouraging encoder output to stay close to codebook.
        """
        x = self.node_emb.weight
        # nn.Embedding is excluded from AMP autocast per PyTorch policy, so the
        # (num_edges, in_channels) gather inside SAGEConv would stay fp32 —
        # doubling GPU memory. Cast to the active autocast dtype (bf16 or fp16)
        # so the gather runs in 16-bit.
        if torch.is_autocast_enabled() and x.device.type == 'cuda':
            x = x.to(torch.get_autocast_gpu_dtype())

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            if radii is not None:
                radii_t = radii.to(device=x.device, dtype=x.dtype)
                if radii_t.dim() == 1:
                    if radii_t.size(0) != x.size(0):
                        raise ValueError(
                            f"radii length {radii_t.size(0)} must match "
                            f"number of nodes {x.size(0)}"
                        )
                    radii_t = radii_t.unsqueeze(1)
                elif radii_t.dim() == 2:
                    if radii_t.size(0) != x.size(0):
                        raise ValueError(
                            f"radii rows {radii_t.size(0)} must match "
                            f"number of nodes {x.size(0)}"
                        )
                else:
                    raise ValueError("radii must be a 1D or 2D tensor")

                # Per-node Gaussian noise, broadcast across feature channels.
                x = x + torch.randn_like(x) * radii_t
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        # Lake-identity conditioning: shift all gene representations by a
        # learned per-lake vector before VQ discretization.  This lets the
        # VQ layer assign different codes to the same gene across lakes.
        if lake_idx is not None and self.lake_emb is not None:
            if not isinstance(lake_idx, torch.Tensor):
                lake_idx = torch.tensor(lake_idx, device=x.device)
            lake_shift = self.lake_proj(self.lake_emb(lake_idx))  # (codebook_channels,)
            x = x + lake_shift.unsqueeze(0)                        # broadcast over genes

        quantized, indices, commit_loss = self.vq(x)
        decoded = self.decoder(quantized)

        return x, decoded, quantized, indices, commit_loss

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def reconstruction_loss(self, decoded, target_adj, batch_size=1024):
        """
        Mean-squared error between decoded·decoded.T and the target adjacency,
        evaluated in blocks to keep GPU memory bounded at O(batch_size × N).

        The inner-product matrix is passed through a sigmoid to map from
        unconstrained real values to [0,1], matching the adjacency's [0,1] scale.
        This replaces the original per-block min-max scaling which caused
        inconsistent normalization across blocks (each block got its own min/max).

        Parameters
        ----------
        decoded : torch.Tensor (N, D)
            Decoded node representations.
        target_adj : torch.Tensor (N, N)
            Dense target adjacency matrix. Values in [0, 1].
        batch_size : int
            Number of rows to process at once.

        Returns
        -------
        mse : torch.Tensor (scalar)
            Mean squared error.
        """
        N = decoded.size(0)
        device = decoded.device

        # Move target_adj to GPU in blocks to avoid a single 3+ GiB allocation
        # that fragments CUDA memory when combined with autograd intermediates.
        losses = []
        n_entries = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            block = decoded[start:end]                     # (B, D)
            recon = block @ decoded.T                       # (B, N)
            recon = torch.sigmoid(recon)                    # Map to [0, 1]
            tgt = target_adj[start:end].to(
                device=device, dtype=torch.float32)         # (B, N) on GPU

            losses.append((recon - tgt).pow(2).sum())
            n_entries += recon.numel()

        return torch.stack(losses).sum() / n_entries

    # ------------------------------------------------------------------
    # Training loops
    # ------------------------------------------------------------------

    def train_joint(self, model_save_path, list_of_edge_indices,
                    list_of_target_adjs, epochs=5, lr=1e-4,
                    commit_alpha=0.25, radii=None, lake_ids=None):
        """
        Joint training with lake-identity conditioning.

        Shuffles lake order each epoch and conditions
        the VQ discretization on a learned per-lake embedding (see ``forward``).
        This addresses two problems with sequential training:

        1. **Order effects** — in sequential training, earlier lakes influence
           the codebook more than later lakes.  Shuffling each epoch gives every
           lake equal opportunity to shape the shared codebook.
        2. **Lake-blind representations** — without conditioning, the model
           forces every lake into the same codebook with no way to express
           "this gene behaves unusually in Lake X".  The lake embedding shifts
           gene representations before VQ, so the same gene can land in
           different codes depending on lake context.

        The lake embeddings are learned parameters updated via the same
        optimizer as the rest of the model.  They capture lake-specific
        structure while the shared SAGEConv + VQ codebook captures gene
        programs that generalise across lakes.

        Requires ``n_lakes`` to be set at construction time.

        Parameters
        ----------
        model_save_path : str
            Path to save trained model weights.
        list_of_edge_indices : list[torch.Tensor]
            Edge indices (2 × n_edges) for each graph.
        list_of_target_adjs : list[torch.Tensor]
            Dense target adjacency matrices (N × N, values in [0, 1]).
        epochs : int
            Number of full passes over all lakes.
        lr : float
            Learning rate for Adam.
        commit_alpha : float
            Weight for the VQ commitment loss term.
        radii : list[np.ndarray] or None
            Per-node noise radii for each graph.
        lake_ids : list[int] or None
            Integer ID for each graph (0..n_lakes-1).  If None, uses the
            graph index as the lake ID (one lake per graph).  Multiple graphs
            CAN share the same lake ID (e.g. different years of the same lake).
        """
        if self.lake_emb is None:
            raise RuntimeError(
                "train_joint requires n_lakes to be set at construction time. "
                "Use embedder.train() (sequential) for models without lake "
                "conditioning."
            )

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        device = next(self.parameters()).device
        use_amp = (device.type == 'cuda')
        if use_amp:
            print("[train_joint] Using AMP bf16 autocast (no GradScaler needed)")
        n_graphs = len(list_of_edge_indices)

        if lake_ids is None:
            lake_ids = list(range(n_graphs))

        n_lakes_unique = len(set(lake_ids))
        print(f"[train_joint] device={device}, "
              f"cuda_available={torch.cuda.is_available()}, "
              f"cuda_device_count={torch.cuda.device_count()}")
        print(f"[train_joint] {n_graphs} graphs from {n_lakes_unique} unique lakes, "
              f"{epochs} epochs, lr={lr}, commit_alpha={commit_alpha}")

        # Best-epoch tracking: checkpoint the epoch with the lowest
        # mean(edge_loss) + 0.1 * mean(commit_loss) — reconstruction quality
        # with a light VQ/codebook signal, not the final epoch.
        best = {'state': None, 'crit': float('inf'),
                'commit': None, 'edge': None, 'epoch': -1}

        for ep in range(epochs):
            # Shuffle lake order so no lake is systematically advantaged
            perm = np.random.permutation(n_graphs)
            ep_losses = []
            ep_edge_losses = []
            ep_commit_losses = []

            pbar = tqdm(perm, desc=f"Epoch {ep+1}/{epochs}", unit="graph",
                        leave=False)
            for g_idx in pbar:
                edge_index = list_of_edge_indices[g_idx].to(device)
                target_adj = list_of_target_adjs[g_idx]
                lake_idx = lake_ids[g_idx]

                r = None
                if radii is not None:
                    r = torch.tensor(radii[g_idx], dtype=torch.float32,
                                     device=device)

                # bf16 autocast: same range as fp32 (no GradScaler needed),
                # half the memory for the (E, in_channels) SAGEConv gather.
                with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=use_amp):
                    _, decoded, _, _, commit_loss = self.forward(
                        edge_index, radii=r, lake_idx=lake_idx,
                    )
                    edge_loss = self.reconstruction_loss(decoded, target_adj)
                loss = edge_loss + commit_alpha * commit_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ep_losses.append(loss.item())
                ep_edge_losses.append(edge_loss.item())
                ep_commit_losses.append(commit_loss.item())

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    edge=f"{edge_loss.item():.4f}",
                    commit=f"{commit_loss.item():.4f}",
                )
                # Note: do NOT call torch.cuda.empty_cache() here — it forces a
                # device sync and defeats the caching allocator on every graph,
                # slowing training 1.5-3x with no peak-memory benefit.

            # Track the best epoch by edge + 0.1*commit (means over graphs).
            if ep_edge_losses:
                epoch_edge = float(np.mean(ep_edge_losses))
                epoch_commit = float(np.mean(ep_commit_losses))
                epoch_crit = epoch_edge + 0.1 * epoch_commit
                if epoch_crit < best['crit']:
                    best = {
                        'state': {k: v.detach().cpu().clone()
                                  for k, v in self.state_dict().items()},
                        'crit': epoch_crit,
                        'edge': epoch_edge,
                        'commit': epoch_commit,
                        'epoch': ep + 1,
                    }
            best_str = (f" | best ep {best['epoch']} {best['crit']:.4f}"
                        if best['state'] is not None else "")
            print(f"Epoch {ep+1}/{epochs} summary | "
                  f"loss mean={np.mean(ep_losses):.4f} "
                  f"min={np.min(ep_losses):.4f} "
                  f"max={np.max(ep_losses):.4f} | "
                  f"edge={np.mean(ep_edge_losses):.4f} "
                  f"commit={np.mean(ep_commit_losses):.4f}{best_str}")

        # Restore the best epoch into the model and save it as the checkpoint,
        # so downstream steps reflect the best epoch rather than the final one.
        if best['state'] is not None:
            self.load_state_dict(best['state'])
        torch.save(self.state_dict(), model_save_path)
        # Save best-epoch losses for diagnostics
        final_losses = {
            'edge_loss': best['edge'] if best['state'] is not None
            else float(np.mean(ep_edge_losses)),
            'commit_loss': best['commit'] if best['state'] is not None
            else float(np.mean(ep_commit_losses)),
            'best_epoch': best['epoch'] if best['state'] is not None else epochs,
        }
        loss_path = model_save_path.replace('.pt', '_loss.pkl')
        torch.save(final_losses, loss_path)
        print(f"[train_joint] Best epoch {final_losses['best_epoch']} "
              f"(edge_loss={final_losses['edge_loss']:.4f}, "
              f"commit_loss={final_losses['commit_loss']:.4f}) saved to "
              f"{model_save_path}")

    # ------------------------------------------------------------------
    # Embedding extraction (for downstream analysis)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_vq_assignments(self, edge_index, lake_idx=None):
        """
        Get VQ code assignments for all genes given a graph.

        Parameters
        ----------
        edge_index : torch.Tensor (2, n_edges)
        lake_idx : int, optional
            Lake identity for conditioning (used when model was trained jointly).

        Returns
        -------
        indices : torch.Tensor (n_nodes,)
            Codebook index (0..codebook_size-1) for each gene.
        """
        self.eval()
        _, _, _, indices, _ = self.forward(edge_index, lake_idx=lake_idx)
        return indices

    @torch.no_grad()
    def get_codebook_histogram(self, edge_index, lake_idx=None):
        """
        Get normalized VQ code histogram for a graph — this is the
        "lake embedding": a probability distribution over discrete codes.

        Parameters
        ----------
        edge_index : torch.Tensor (2, n_edges)
        lake_idx : int, optional
            Lake identity for conditioning (used when model was trained jointly).

        Returns
        -------
        hist : torch.Tensor (codebook_size,)
            Normalized histogram summing to 1.
        """
        indices = self.get_vq_assignments(edge_index, lake_idx=lake_idx).cpu()
        hist = torch.bincount(indices, minlength=self.vq.codebook_size).float()
        hist = hist / hist.sum()
        return hist
