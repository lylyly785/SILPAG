# SILPAG

**Spatially-Informed Identification and Localization of Perturbation-Altered Genes**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Beyond Differential Expression: Identifying Condition-Responsive Genes and Tissue Regions through Cross-Condition Spatial Transcriptomics**

SILPAG is a computational framework for identifying **condition-responsive genes (CRGs)** and **condition-responsive tissue regions (CRRs)** from cross-condition spatial transcriptomics data. Unlike traditional differential expression analysis, SILPAG leverages spatial gene expression patterns to detect genes and regions that are altered between biological conditions (e.g., disease vs. healthy, treated vs. untreated).

## Framework Overview

<div align="center">
<img src="/docs/images/Overview.png" width="90%">
</div>

## Key Features

- **Spatial Gene Modeling (SGM):** Encodes each gene's spatial expression pattern into a compact representation via a Vision Transformer autoencoder with vector quantization.
- **Anchor-Gene-Guided Optimal Transport (AGOT):** Aligns codebook representations across conditions using Gromov-Wasserstein optimal transport, guided by anchor (housekeeping) genes.
- **CRG Identification:** Detects condition-responsive genes by comparing view-transferred and observed spatial gene maps, with multiple null models (z-score, gamma, percentile, top-k, fold-change).
- **Alteration Type Disentanglement:** Classifies CRGs into three alteration types — differential abundance, differential prevalence, and differential spatial pattern.
- **CRR Delineation:** Identifies condition-responsive tissue regions via marker-based (Otsu thresholding on SDM) or marker-free (Leiden clustering on discrepancy vectors) approaches.
- **Histology Integration:** Optionally incorporates H&E histology image features (via HIPT) through cross-attention fusion.
- **Pathway Enrichment:** Built-in GO/KEGG/Reactome enrichment analysis and publication-ready bubble plots.

## Installation

```bash
git clone https://github.com/lylyly785/SILPAG.git
cd SILPAG
pip install -r requirements.txt
```

### Dependencies


| Package | Version |
|---------|---------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0 |
| scanpy | ≥ 1.12 |
| anndata | ≥ 0.12 |
| scipy | ≥ 1.17 |
| scikit-learn | ≥ 1.8 |
| einops | ≥ 0.8 |
| entmax | ≥ 1.3 |
| gseapy | (for pathway enrichment) |
| rpy2 | ≥ 3.5 (optional, for SPARK-X) |

See `requirements.txt` for the full list.

## Quick Start

```python
import SILPAG as sp
import scanpy as sc
import numpy as np

# ---- 1. Load data ----
adata_ref = sc.read_h5ad('path/to/reference.h5ad')  # e.g., healthy
adata_tgt = sc.read_h5ad('path/to/target.h5ad')      # e.g., disease

# ---- 2. Preprocessing ----
sp.prefilter_specialgenes(adata_ref)
sp.prefilter_specialgenes(adata_tgt)

# ---- 3. Configure and train ----
config = sp.Config(
    num_slice=2,
    device='cuda',
    train=True,
    save=True,
    model_path='saved_model/my_model',
)
sp.set_seed(42)

# Prepare gene spatial images and dataset
# (gene_images: list of arrays, shape [n_genes, H, W] per slice)
dataset = sp.GeneDataset(gene_images, labels=None, args=config)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

# Train the model (Stage I: pretrain + Stage II: AGOT alignment)
model, dataset, PI, Q = sp.train_marker(train_loader, hist_data=None, args=config)

# ---- 4. Extract gene embeddings and codes ----
embeddings = sp.get_embedding(model, dataset, args=config)
code = sp.get_code(model, dataset, pi_init=PI, source_idx=0, target_idx=1, args=config)

# ---- 5. Identify CRGs ----
# Generate view-transferred reconstruction
result_ref = model.generate(config, adata_tgt, tgt_index=1, code_id='gene_code')

scores = sp.crg.compute_crg_scores(result_ref, adata_tgt)
crg_result = sp.crg.identify_crgs(scores, anchor_mask=dataset.labels, p_threshold=0.05)
print(f"Identified {crg_result['is_crg'].sum()} CRGs")

# ---- 6. Disentangle alteration types ----
alteration_df = sp.crg.disentangle_alteration_types(result_ref, adata_tgt, crg_result['is_crg'])
print(alteration_df['alteration_type'].value_counts())

# ---- 7. Delineate CRRs ----
# Marker-based
crr = sp.crr.delineate_crr_marker(result_ref, adata_tgt, gene='GENE_NAME', sigma=1.0)
sp.crr.plot_crr(crr['coords'], crr['crr_mask'], sdm_smoothed=crr['sdm_smoothed'])

# Marker-free
crr_free = sp.crr.delineate_crr_markerfree(result_ref, adata_tgt, crg_result['is_crg'])
sp.crr.plot_crr(crr_free['coords'], crr_free['crr_mask'], title='Marker-free CRR')

# ---- 8. Pathway enrichment ----
crg_genes = adata_tgt.var_names[crg_result['is_crg']].tolist()
enr_df = sp.crg.run_pathway_enrichment(crg_genes, gene_sets='KEGG_2021_Human')
sp.crg.plot_pathway_bubble(enr_df, title='CRG Pathway Enrichment')
```

## Project Structure

```
SILPAG/
├── __init__.py                 # Public API
├── main.py                     # Training pipeline, dataset, inference
├── model.py                    # SILPAG model (ViT encoder, VQ codebook, decoder)
├── agot.py                     # Anchor-Gene-Guided Optimal Transport
├── sinkhorn.py                 # Sinkhorn algorithm for OT
├── util.py                     # Utilities (preprocessing, clustering, spatial smoothing)
├── extract_hist_feature.py     # Histology feature extraction (HIPT)
├── hipt_4k.py                  # HIPT 4K model
├── hipt_model_utils.py         # HIPT utilities
├── vision_transformer.py       # ViT backbone (256×256)
├── vision_transformer4k.py     # ViT backbone (4096×4096)
├── crg/                        # CRG identification module
│   ├── identification.py       # CRG scoring and statistical testing
│   ├── disentangle.py          # Alteration type classification
│   └── go_analysis.py          # Pathway/GO enrichment and visualization
├── crr/                        # CRR delineation module
│   └── delineation.py          # Marker-based and marker-free CRR methods
├── data/                       # Demo data (download separately)
└── saved_model/                # Pre-trained model checkpoints
```

## Demo Data

The demo uses a human breast cancer (hBC) cross-condition case study with two spatial transcriptomics slices. Due to GitHub file size limits, large files need to be downloaded separately and placed into the corresponding directories.

**Download:** [OneDrive](https://cuhko365-my.sharepoint.com/:f:/g/personal/225040459_link_cuhk_edu_cn/IgB8oVwMOxjRTZndM5LLiexwASgi3ts3kse0_dRJZ9SLnzc?e=XsNsLK)

> ⚠️ After downloading, place the files as follows:
> ```
> SILPAG/data/H1_nb.h5ad
> SILPAG/data/V02_nb.h5ad
> SILPAG/saved_model/hBC_demo.pkl
> ```

### Data Files

| File | Size | Format | Description |
|------|------|--------|-------------|
| `H1_nb.h5ad` | 328 MB | AnnData (h5ad) | Breast cancer slice (target condition) |
| `V02_nb.h5ad` | 456 MB | AnnData (h5ad) | Healthy breast slice (reference condition) |
| `hBC_demo.pkl` | 17 MB | PyTorch state dict | Pre-trained SILPAG model weights |
| `hBC_demo.json` | 360 KB | JSON | Model configuration / hyperparameters |

### h5ad File Structure

Each `.h5ad` file is an [AnnData](https://anndata.readthedocs.io/) object with the following structure:

```
AnnData object
├── X                     # Gene expression matrix (n_spots × n_genes), float32
│                         #   H1_nb: (613, 7874) dense
│                         #   V02_nb: (1896, 7874) sparse (CSR)
├── obs                   # Spot-level metadata
│   ├── array_x, array_y  # Array coordinates (for gene image indexing)
│   ├── cell_type          # Annotated cell/tissue type
│   └── disease            # Condition label
├── var                   # Gene-level metadata
│   ├── hkgene             # Housekeeping gene flag (1 = anchor gene)
│   └── n_cells            # Number of expressing spots
├── obsm
│   └── spatial            # Spatial coordinates (n_spots × 2)
├── varm
│   └── gene_img           # Pre-computed spatial gene images (n_genes × H × W)
│                          #   H1_nb: (7874, 26, 30)
│                          #   V02_nb: (7874, 61, 92)
└── uns
    └── spatial            # H&E histology image and scale factors
        └── images/hires   # High-resolution H&E image (uint8, RGB)
```

**H1_nb (target):** 613 spots, 7874 genes, 7 tissue types (adipose tissue, breast glands, cancer in situ, connective tissue, immune infiltrate, invasive cancer, undetermined)

**V02_nb (reference):** 1896 spots, 7874 genes, 9 cell types (fibroblast, B cell, pericyte, endothelial cell, luminal epithelial cell, mammary gland epithelial cell, adipocyte, etc.)

## Method Details

SILPAG operates in two stages:

**Stage I — Spatial Gene Modeling (SGM):**
Each gene's 2D spatial expression pattern is encoded by a patch-based Vision Transformer. A learnable codebook with sparse vector quantization (via entmax) captures discrete spatial expression archetypes. An auxiliary decoder reconstructs spatial ranking patterns and distribution parameters.

**Stage II — Cross-Condition Alignment via AGOT:**
Anchor genes (housekeeping genes stable across conditions) guide a Gromov-Wasserstein optimal transport between codebooks of different conditions. An EM algorithm iteratively refines gene-level condition-responsiveness probabilities and the inter-codebook transport plan.

**Downstream Analysis:**
- CRG scores are computed as the Frobenius norm between view-transferred and observed spatial gene maps.
- Alteration types are disentangled via KS tests (abundance), Fisher's exact tests (prevalence), and Moran's I (spatial pattern).
- CRRs are delineated by thresholding spatial discrepancy maps or clustering spot-level discrepancy vectors.

## Citation

If you use SILPAG in your research, please cite:

```bibtex
@article{silpag2025,
  title={Beyond Differential Expression: Identifying Condition-Responsive Genes and Tissue Regions through Cross-Condition Spatial Transcriptomics},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an [issue](https://github.com/lylyly785/SILPAG/issues) on GitHub.
