"""
CRR (Condition-Responsive Region) Delineation.

Two modes:
  1. Marker-based  — Given a known condition-associated gene g, compute the SDM
     D_g = |x_ref->tgt_g - x_tgt_g|, smooth with Gaussian kernel, then threshold
     via Otsu's method to partition spots into CRR vs non-CRR.
  2. Marker-free   — Use all identified CRGs. Each spot gets a discrepancy vector
     f_i ∈ R^|CRGs|. Leiden clustering on {f_i} yields two spatial domains;
     the one with higher aggregate discrepancy is the CRR.
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's method on a 1-D array of continuous values."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    best_tau, best_var = sorted_vals[0], -1.0
    for i in range(1, n):
        w0 = i / n
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = sorted_vals[:i].mean()
        mu1 = sorted_vals[i:].mean()
        between_var = w0 * w1 * (mu0 - mu1) ** 2
        if between_var > best_var:
            best_var = between_var
            best_tau = sorted_vals[i]
    return best_tau


def _smooth_sdm_on_grid(sdm: np.ndarray, coords: np.ndarray, sigma: float = 1.0):
    """
    Smooth SDM values on a spatial grid using Gaussian filter.

    For irregular spot layouts, we rasterise onto a grid, smooth, then
    read back the values at spot positions.
    """
    # Build a grid mapping
    x, y = coords[:, 0], coords[:, 1]
    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    # Map coordinates to grid indices
    x_idx = np.searchsorted(x_unique, x)
    y_idx = np.searchsorted(y_unique, y)

    grid = np.zeros((len(y_unique), len(x_unique)), dtype=np.float64)
    grid_count = np.zeros_like(grid)
    for i in range(len(sdm)):
        grid[y_idx[i], x_idx[i]] += sdm[i]
        grid_count[y_idx[i], x_idx[i]] += 1.0
    grid_count[grid_count == 0] = 1.0
    grid /= grid_count

    # Gaussian smooth
    smoothed_grid = gaussian_filter(grid, sigma=sigma)

    # Read back
    smoothed = np.array([smoothed_grid[y_idx[i], x_idx[i]] for i in range(len(sdm))])
    return smoothed


# ---------------------------------------------------------------------------
# Marker-based CRR delineation
# ---------------------------------------------------------------------------

def delineate_crr_marker(
    result_ref: ad.AnnData,
    adata: ad.AnnData,
    gene: str,
    sigma: float = 1.0,
) -> dict:
    """
    Marker-based CRR delineation (Eq. 31-32 in paper).

    Parameters
    ----------
    result_ref : AnnData
        View-transferred reconstruction.
    adata : AnnData
        Original target AnnData with obsm['spatial'].
    gene : str
        Condition-associated marker gene name.
    sigma : float
        Gaussian smoothing sigma.

    Returns
    -------
    dict with keys:
        'gene', 'sdm', 'sdm_smoothed', 'threshold', 'crr_mask', 'coords'
    """
    gene_idx = list(adata.var_names).index(gene)
    X_ref = _to_dense(result_ref.X)
    X_tgt = _to_dense(adata.X)

    # SDM: D_g = |x_ref->tgt_g - x_tgt_g|
    sdm = np.abs(X_ref[:, gene_idx] - X_tgt[:, gene_idx])

    coords = np.array(adata.obsm['spatial'])
    sdm_smoothed = _smooth_sdm_on_grid(sdm, coords, sigma=sigma)

    # Otsu threshold
    tau = _otsu_threshold(sdm_smoothed)
    crr_mask = sdm_smoothed > tau

    return {
        'gene': gene,
        'sdm': sdm,
        'sdm_smoothed': sdm_smoothed,
        'threshold': float(tau),
        'crr_mask': crr_mask,
        'coords': coords,
    }


# ---------------------------------------------------------------------------
# Marker-free CRR delineation
# ---------------------------------------------------------------------------

def delineate_crr_markerfree(
    result_ref: ad.AnnData,
    adata: ad.AnnData,
    crg_mask: np.ndarray,
    resolution: float = 0.5,
    n_neighbors: int = 15,
) -> dict:
    """
    Marker-free CRR delineation via Leiden clustering (Eq. 33 in paper).

    Parameters
    ----------
    result_ref : AnnData
        View-transferred reconstruction.
    adata : AnnData
        Original target AnnData with obsm['spatial'].
    crg_mask : np.ndarray of bool
        Which genes are CRGs.
    resolution : float
        Leiden clustering resolution.
    n_neighbors : int
        Number of neighbours for the spot-level KNN graph.

    Returns
    -------
    dict with keys:
        'crr_mask', 'cluster_labels', 'aggregate_scores', 'coords'
    """
    import scanpy as sc

    crg_indices = np.where(np.asarray(crg_mask, dtype=bool))[0]
    X_ref = _to_dense(result_ref.X)
    X_tgt = _to_dense(adata.X)

    # Discrepancy vectors f_i for each spot
    diff = np.abs(X_ref - X_tgt)
    F = diff[:, crg_indices]  # (n_spots, n_crgs)

    # Build a temporary AnnData for Leiden clustering
    adata_tmp = ad.AnnData(X=F)
    adata_tmp.obsm['spatial'] = np.array(adata.obsm['spatial'])

    sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.leiden(adata_tmp, resolution=resolution, key_added='leiden')

    labels = adata_tmp.obs['leiden'].values.astype(str)
    unique_labels = np.unique(labels)

    # Compute aggregate discrepancy per cluster
    agg = {}
    for lab in unique_labels:
        mask = labels == lab
        agg[lab] = np.mean(np.sum(np.abs(F[mask]), axis=1))

    # CRR = cluster with highest aggregate discrepancy
    crr_label = max(agg, key=agg.get)
    crr_mask = labels == crr_label

    return {
        'crr_mask': crr_mask,
        'cluster_labels': labels,
        'aggregate_scores': agg,
        'coords': np.array(adata.obsm['spatial']),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_crr(
    coords: np.ndarray,
    crr_mask: np.ndarray,
    title: str = 'CRR Delineation',
    sdm_smoothed: np.ndarray = None,
    figsize: tuple = (14, 5),
    save_path: str = None,
):
    """
    Visualise CRR delineation results.

    If sdm_smoothed is provided, shows two panels:
      left  = smoothed SDM heatmap
      right = CRR binary map
    Otherwise shows only the CRR binary map.
    """
    n_panels = 2 if sdm_smoothed is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0
    if sdm_smoothed is not None:
        ax = axes[panel_idx]
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=sdm_smoothed, cmap='Reds', s=12, edgecolor='none',
        )
        plt.colorbar(sc, ax=ax, label='Smoothed SDM')
        ax.set_title('Smoothed SDM', fontsize=14)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        panel_idx += 1

    ax = axes[panel_idx]
    colors = np.where(crr_mask, 'tomato', 'lightgrey')
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=12, edgecolor='none')
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
               markersize=8, label='CRR'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey',
               markersize=8, label='Non-CRR'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
