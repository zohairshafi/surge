# SURGE: Spectral Uncertainty‑aware Representation of Gene Co‑expression Networks across Time, Space, and Environment

Gene co-expression networks capture transcriptional coordination across specimens, revealing functional modules and regulatory architecture in biological systems. The dominant analytical paradigm - Weighted Gene Co-expression Network Analysis (WGCNA) - constructs independent, soft-thresholded networks for each condition and identifies discrete gene modules. This framework suffers from three fundamental limitations: (1) edge weights are filtered by an arbitrary threshold, discarding weak but biologically meaningful signals (2) networks built separately for each condition cannot be directly compared, precluding quantitative analysis of co-expression change across time, space, or treatment and (3) module membership is static within each condition, obscuring the continuous, dynamic nature of regulatory network rewiring. Here we present \textsc{SURGE} - Spectral Uncertainty-aware Representation of Gene Co-expression Networks, a deep learning method that reframes co-expression analysis as a graph representation learning problem, enabling a shift from independent module discovery per condition to learned, comparable embeddings across all conditions via a shared discrete codebook. SURGE uses spectral decomposition to construct co-expression graphs at multiple resolutions without thresholding, quantifies per-gene edge uncertainty for data-driven regularization, and employs a vector-quantized graph neural network (VQ-GNN) to compress each network into a low-dimensional histogram over a shared codebook. Because the codebook is shared across all networks, these histograms are directly comparable, enabling quantitative tracking of co-expression evolution over time and across populations. We validate SURGE using an experiment spanning 381 head kidney transcriptomes from 16 threespine stickleback (Gasterosteus aculeatus) populations sampled annually from 2019-2024, including eight long-established source lakes and eight recently colonized recipient lakes created by experimental transplantation. We find that co-expression network architecture diverges significantly over time following colonization. Recipient lakes exhibit faster network changes than source lakes, consistent with ongoing regulatory evolution in newly founded populations. Our findings align with independent experimental studies showing that SURGE behaves as a complementary method to well established analysis pipelines, while also offering unique advantages by providing a shared embedding space for quantitative network comparison across time and populations.
## Installation

```bash
pip install -r requirements.txt
```

**Core dependencies:** PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.4, vector-quantize-pytorch ≥ 1.14.

## Architecture

```text
                   ┌──────────────────────────────────────────────┐
Expression ───────►│  CoexpressionGraphBuilder                    │
matrices           │  Spectral decomposition + multi-scale        │
                   │  reconstruction → adjacency matrices          │
                   └──────────────┬───────────────────────────────┘
                                  │ edge_index, radii
                   ┌──────────────▼───────────────────────────────┐
                   │  VQGNN                                       │
                   │  3× SAGEConv → VQ codebook (100 codes)       │
                   │  → Sigmoid decoder → loss vs. ground truth    │
                   └──────────────┬───────────────────────────────┘
                                  │ codebook histograms (100-dim)
                   ┌──────────────▼───────────────────────────────┐
                   │  LakeAnalyzer                                │
                   │  PCA · Wasserstein · PERMANOVA · clustering  │
                   │  Label permutation · Code enrichment          │
                   └──────────────────────────────────────────────┘
```

## Module Reference

### `data.py` — `SticklebackData`

Loads and manages the stickleback transcriptome dataset.

```python
from surge import SticklebackData

data = SticklebackData(
    transcriptome_path="data/7.HKTranscriptome.csv",
    metadata_path="data/1.Metadata.csv",
    morphology_path="data/2.Morphology.csv",
    input_scale="log-cpm",       # log-CPM normalization
    log1p=True,                  # log(1+x) transform
    prefilter_top_n=None,        # or 10000 for 10k-gene subset
)
```

| Method | Description |
|--------|-------------|
| `stratify(by='year_lake', sex=None, infection=None)` | Split fish into strata. `by` can be `'year_lake'`, `'sex_year_lake'`, `'infection_year_lake'`, or `'lake'`. |
| `build_all_matrices(by='year_lake')` | Build expression matrices for all strata. Returns `dict[key, np.ndarray]` of shape `(n_fish, n_genes)`. |
| `get_expression_matrix(lake=..., year=..., sex=..., infection=...)` | Subset expression data with flexible filtering. |
| `get_lake_role(lake)` | Returns `'Source'` or `'Recipient'`. |
| `get_lake_ecotype(lake)` | Returns `'Benthic'` or `'Limnetic'`. |
| `get_genotype(lake)` | Returns genotype pool assignment. |

Key attributes: `metadata` (DataFrame), `gene_names` (list), `fish_to_lake`, `fish_to_sex`, `fish_to_infection`, `lake_to_genotype`, `years`.

---

### `graphs.py` — `CoexpressionGraphBuilder`

Builds co-expression graphs via spectral decomposition of the correlation matrix.

```python
from surge import CoexpressionGraphBuilder

builder = CoexpressionGraphBuilder(
    n_eigencomponents=64,                    # eigenvectors to retain
    reconstruction_levels=[2, 4, 16, 32],    # multi-scale reconstruction
    target_density=0.05,                     # sparsification target
    k_graph=15,                              # k-NN for graph construction
)

graphs = builder.build_all(matrix_dict, output_dir="output/")
```

| Method | Description |
|--------|-------------|
| `build(expression_matrix)` | Full pipeline: z-score → correlation → eigendecomposition → multi-scale reconstruction → adjacency. |
| `build_fast(expression_matrix)` | Faster alternative using direct spectral approximation. |
| `build_all(matrix_dict, output_dir, keep_in_memory)` | Build graphs for all matrices in a dict. Caches to disk via manifest; reuses cached graphs on subsequent runs. |

Each graph entry in the returned dict contains `edge_index` (PyG format), `radii` (residual norms per node), and metadata. Degenerate graphs (too few fish or near-zero variance) are flagged via `is_degenerate()` and assigned uniform embeddings.

**Graph caching:** Graphs are pickled individually to `output_dir/graphs/` with a manifest file (`graphs_manifest.pkl`) tracking which strata map to which graph file. The `radii.pkl` file stores per-gene radii for multi-scale reconstruction. If all three files exist, `build_all` skips recomputation.

---

### `vqgnn.py` — `VQGNN`

Vector-Quantized Graph Neural Network. The core model.

```python
from surge import VQGNN

model = VQGNN(
    n_nodes=22729,           # number of genes
    in_channels=64,          # input feature dim
    hidden_channels=64,      # SAGEConv hidden dim
    out_channels=16,         # final node embedding dim
    num_layers=3,            # SAGEConv layers
    dropout=0.2,
    codebook_size=100,       # number of VQ codes
    codebook_dim=16,         # codebook entry dimension
    decoder_channels=64,     # sigmoid decoder hidden dim
    n_lakes=None,            # None = sequential (no lake conditioning)
    commit_alpha=0.25,       # commitment loss weight
)
```

**Two training paradigms:**

| Paradigm | `n_lakes` | Behavior |
|----------|-----------|----------|
| **Sequential** | `None` | No lake-specific conditioning. Graph order shuffled each epoch. Same gene → same VQ code regardless of which graph it appears in. |
| **Joint** | `17` (or N lakes) | Lake embeddings concatenated to node features. All graphs trained simultaneously. Same gene can map to different codes across lakes. |

| Method | Description |
|--------|-------------|
| `forward(edge_index, radii, lake_idx)` | Encode nodes → VQ quantize → decode → reconstructed adjacency. Returns `(decoded_adj, vq_loss, perplexity)`. |
| `reconstruction_loss(decoded, target_adj, batch_size)` | Sigmoid + MSE computed in blocks to manage memory. |
| `train_joint(model_save_path, edge_indices, radii_list, ...)` | Full training loop with per-epoch graph shuffling (sequential) or joint batching. Saves model checkpoint and loss history. |
| `get_vq_assignments(edge_index, lake_idx)` | Returns per-node VQ code indices `(n_nodes,)`. |
| `get_codebook_histogram(edge_index, lake_idx)` | Returns 100-dim histogram of VQ code usage for a graph. |

**Model outputs saved:**
- `model.pt` / `model_last.pt` — model weights
- `model_kwargs.pkl` — constructor arguments
- `model_loss.pkl` — per-epoch loss history
- `model_opt.pt` — optimizer state

---

### `embedder.py` — `LakeEmbedder`

Training harness that wires `SticklebackData` → `CoexpressionGraphBuilder` → `VQGNN` and produces embeddings.

```python
from surge import LakeEmbedder

embedder = LakeEmbedder(model, device="cuda")
embedder.train(graphs_dict, radii_dict,
               save_path="output/model.pt",
               epochs=20, lr=5e-4, joint=False,
               commit_alpha=1.0, shuffle_graphs=True)

embeddings = embedder.embed_all(graphs_dict)
# -> dict[str, np.ndarray]  (key -> 100-dim histogram)
```

| Method | Description |
|--------|-------------|
| `train(graphs_dict, radii_dict, save_path, epochs, lr, joint, commit_alpha, shuffle_graphs)` | Full training loop. Handles graph loading/caching, degenerate graph detection, and per-epoch shuffling. |
| `embed(graph)` | Generate codebook histogram for a single graph. |
| `embed_all(graphs_dict)` | Generate histogram embeddings for all graphs. |
| `build_gene_vq_mappings(graphs_dict)` | Map every gene in every stratum to its VQ code. Returns `(vq_to_gene, gene_to_vq)` dicts and saves `gene_mappings.pkl`. |

---

### `analysis.py` — `LakeAnalyzer` & `GeneNetworkAnalyzer`

Downstream analysis of VQ histogram embeddings.

```python
from surge import LakeAnalyzer

analyzer = LakeAnalyzer(embeddings, data=stickleback_data)
```

| Method | Description |
|--------|-------------|
| `pca_project(n_components=2)` | PCA of 100-dim embeddings. Returns `(projected, keys)`. |
| `wasserstein_temporal(base_year=2019)` | Pairwise Wasserstein distances between all strata years. Returns distance matrix. |
| `compute_temporal_slopes(wasserstein_distances)` | Per-lake linear regression of Wasserstein distance vs. year. Returns slope per lake. |
| `source_vs_recipient_slope_test(wasserstein_distances)` | Mann-Whitney U & t-test comparing source vs. recipient slopes. |
| `hierarchical_clustering(method='ward', n_clusters=None)` | Agglomerative clustering of embeddings. Returns `(Z, cluster_labels, order)`. |
| `analyze_clusters(n_clusters=4)` | Fisher's exact test for enrichment of lake role, ecotype, sex, infection, and year in each cluster. |
| `silhouette_scan(max_k=10)` | Silhouette score for k=2..max_k clusters. |
| `permanova_decomposition(metadata, n_permutations=1000)` | PERMANOVA (marginal R²) for Year, Lake, Lake Category, Ecotype, Ancestry, Sex, Infection. |
| `label_permutation_test(label_type, n_permutations=1000)` | Permutation test comparing true silhouette to null distribution. |
| `infection_code_enrichment(codebook_size=100)` | Mann-Whitney U per VQ code: infected vs. non-infected. Returns FDR-corrected p-values. |
| `sex_code_enrichment(codebook_size=100)` | Same for female vs. male. |
| `role_code_enrichment(codebook_size=100)` | Same for source vs. recipient lake role. |

```python
analyzer = GeneNetworkAnalyzer(vq_to_gene, gene_to_vq, gene_names=gene_names)
```

| Method | Description |
|--------|-------------|
| `build_network(central_gene, max_genes=100)` | Build a co-occurrence network around a central gene. Nodes = genes, edges = frequency of co-assignment to the same VQ code across strata. |

---

### `plotting.py` — `LakePlotter`

Visualization suite for all analysis outputs.

```python
from surge.plotting import LakePlotter
```

| Method | Description |
|--------|-------------|
| `pca_scatter(projected, keys, data)` | PCA scatter with lake coloring, ecotype markers, year transparency. |
| `wasserstein_grid(wasserstein_distances, ncols=4)` | Multi-panel grid of per-lake Wasserstein distance vs. year with regression lines. |
| `gene_network(G, central_gene)` | NetworkX graph visualization of gene co-occurrence networks. |
| `code_distribution(vq_to_gene, gene_to_vq)` | Bar chart of VQ code usage distribution. |
| `silhouette_scan(scores_dicts)` | Silhouette score vs. k with permutation null band. |
| `infection_codes / sex_codes / role_codes(code_results)` | Volcano-style plots of per-code significance. |
| `permanova_bar(permanova_results)` | Grouped bar chart of PERMANOVA R² values. |

---

### `batch_correction.py`

Batch correction pipeline for raw RNA-seq count data. Applies ComBat via pycombat to correct for technical batch effects between sequencing runs.

```python
from surge.batch_correction import batch_correct_hk

batch_correct_hk(
    metadata_path="data/1.Metadata.csv",
    transcriptome_path="data/7.HKTranscriptome.csv",
    output_dir="output/results_batch_corrected/",
    n_top_genes=25000,
)
```

**Pipeline:**
1. Load raw counts, filter to protein-coding genes
2. Log-CPM normalization
3. Batch assignment: Lindsay lab samples (2019–2022) vs. Rogini lab samples (2023–2024)
4. Gene selection: top-N by mean expression across all samples
5. pycombat correction with empirical Bayes
6. Post-correction PVCA validation
7. Output: single `expression_batch_corrected_pipeline.csv`

---

## Typical Workflow

```python
from surge import SticklebackData, CoexpressionGraphBuilder, VQGNN, LakeEmbedder, LakeAnalyzer

# 1. Load data
data = SticklebackData("data/7.HKTranscriptome.csv", "data/1.Metadata.csv",
                       prefilter_top_n=10000)
matrices = data.build_all_matrices(by="year_lake")

# 2. Build graphs
builder = CoexpressionGraphBuilder(n_eigencomponents=64,
                                    reconstruction_levels=[2, 4, 16, 32])
graphs = builder.build_all(matrices, output_dir="output/10k_genes/")

# 3. Train model
model = VQGNN(n_nodes=data.n_genes, n_lakes=None)  # sequential
embedder = LakeEmbedder(model, device="cuda")
embedder.train(graphs, builder.radii, save_path="output/model.pt", epochs=20)

# 4. Generate embeddings
embeddings = embedder.embed_all(graphs)

# 5. Analyze
analyzer = LakeAnalyzer(embeddings, data=data)
permanova = analyzer.permanova_decomposition(data.metadata)
slopes = analyzer.source_vs_recipient_slope_test(
    analyzer.wasserstein_temporal())
```

## Citation

If you use SURGE in your research, please cite the accompanying manuscript.
