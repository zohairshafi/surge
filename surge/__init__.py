"""
SURGE — Shared Unified Representation of Gene co-Expression.

A framework for comparative co-expression network analysis using
Vector-Quantized Graph Neural Networks (VQ-GNN). Learns a shared discrete
codebook across hundreds of co-expression networks and compares them
quantitatively via Wasserstein distance on codebook histogram embeddings.

Designed for the study of threespine stickleback (Gasterosteus aculeatus)
head kidney transcriptomes across space and time.

Modules
-------
data        - SticklebackData: load CSVs, build mappings, stratify expression data
graphs      - CoexpressionGraphBuilder: build gene co-expression networks
vqgnn       - VQGNN: Vector-Quantized Graph Neural Network model
embedder    - LakeEmbedder: train VQGNN and generate VQ-code histogram embeddings
analysis    - LakeAnalyzer, GeneNetworkAnalyzer: PCA, Wasserstein, gene networks
plotting    - Visualization utilities for embeddings and gene networks
batch_correction - ComBat batch correction for raw RNA-seq count data
"""

from .data import SticklebackData
from .graphs import CoexpressionGraphBuilder
from .vqgnn import VQGNN
from .embedder import LakeEmbedder
from .analysis import LakeAnalyzer, GeneNetworkAnalyzer
