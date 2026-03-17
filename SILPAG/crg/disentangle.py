"""
Alteration-type disentanglement for SILPAG-identified CRGs.

Three tests per gene:
  1. Differential Abundance  - KS test + Cohen's d effect-size gate
  2. Differential Prevalence - Fisher's exact test on binarised expression
  3. Differential Spatial Pattern - One-sided Moran's I permutation test on the SDM
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


def _to_dense_col(X, j):
    """Extract column j from X as a 1-D dense array."""
    col = X[:, j]
    if sparse.issparse(col):
        col = col.toarray()
    return np.asarray(col).ravel()


def _build_weight_matrix(coords, n_neighbors=6):
    """Binary spatial KNN weight matrix (symmetric)."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n = coords.shape[0]
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j_idx in indices[i, 1:]:
            W[i, j_idx] = 1.0
            W[j_idx, i] = 1.0
    return W



def _morans_I(x, W):
    """
    Compute Moran's I and a one-sided analytical p-value (positive autocorrelation).

    Uses the normal approximation under the randomisation assumption, which
    produces continuous p-values and avoids the resolution problem of
    permutation tests with small n_permutations.

    Returns (I_obs, p_value).
    """
    n = len(x)
    x_bar = x.mean()
    z = x - x_bar
    denom = np.dot(z, z)
    if denom == 0:
        return 0.0, 1.0  # constant field

    S0 = W.sum()
    I_obs = (n / S0) * (z @ W @ z) / denom

    # Expected value under H0
    E_I = -1.0 / (n - 1)

    # Variance under randomisation assumption
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)
    m2 = denom / n
    m4 = np.sum(z ** 4) / n
    b2 = m4 / (m2 ** 2)  # kurtosis

    num1 = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    num2 = b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    denom_var = (n - 1) * (n - 2) * (n - 3) * S0 ** 2
    Var_I = (num1 - num2) / denom_var - E_I ** 2

    if Var_I <= 0:
        return I_obs, 1.0

    z_score = (I_obs - E_I) / np.sqrt(Var_I)
    # One-sided: positive autocorrelation
    p = 1.0 - stats.norm.cdf(z_score)

    return I_obs, p


def disentangle_alteration_types(
    result_ref,
    adata,
    crg_mask,
    epsilon=1e-3,
    n_neighbors=6,
    fdr_threshold=0.05,
    abundance_effect_size=0.5,
    spatial_I_threshold=0.1,
):
    """
    Disentangle alteration types for identified CRGs.

    Parameters
    ----------
    result_ref : AnnData
        View-transferred reconstruction (code_ref).
    adata : AnnData
        Original target AnnData with obsm['spatial'] coordinates.
    crg_mask : array-like of bool, shape (n_genes,)
        Which genes to test (True = CRG).
    epsilon : float
        Binarisation threshold for prevalence test.
    n_neighbors : int
        Number of spatial neighbours for Moran's I.
    fdr_threshold : float
        Significance cutoff applied per test type.
    abundance_effect_size : float
        Minimum |Cohen's d| required for abundance significance (default 0.5 = medium).
    spatial_I_threshold : float
        Minimum Moran's I value required for spatial significance (default 0.1).

    Returns
    -------
    df : pd.DataFrame indexed by gene name, with columns:
        'abundance_pval', 'abundance_fdr', 'abundance_cohens_d', 'is_diff_abundance',
        'prevalence_pval', 'prevalence_fdr', 'is_diff_prevalence',
        'spatial_pval', 'spatial_fdr', 'spatial_morans_I', 'is_diff_spatial',
        'alteration_type'
    """
    crg_mask = np.asarray(crg_mask, dtype=bool)
    gene_names = np.asarray(result_ref.var_names)
    crg_indices = np.where(crg_mask)[0]

    X_ref = result_ref.X
    X_tgt = adata.X
    if sparse.issparse(X_ref):
        X_ref = X_ref.toarray()
    if sparse.issparse(X_tgt):
        X_tgt = X_tgt.toarray()

    coords = adata.obsm['spatial']
    W = _build_weight_matrix(coords, n_neighbors=n_neighbors)

    records = []
    for j in crg_indices:
        x_ref = X_ref[:, j].astype(np.float64)
        x_tgt = X_tgt[:, j].astype(np.float64)

        # 1. Differential Abundance (KS test + Cohen's d effect-size gate)
        stat_a, p_a = stats.ks_2samp(x_ref, x_tgt)
        # Cohen's d: pooled standard deviation
        n_ref, n_tgt = len(x_ref), len(x_tgt)
        pooled_std = np.sqrt(
            ((n_ref - 1) * x_ref.var(ddof=1) + (n_tgt - 1) * x_tgt.var(ddof=1))
            / (n_ref + n_tgt - 2)
        )
        cohens_d = (x_ref.mean() - x_tgt.mean()) / (pooled_std + 1e-12)

        # 2. Differential Prevalence (Fisher's exact)
        b_ref = (x_ref > epsilon).astype(int)
        b_tgt = (x_tgt > epsilon).astype(int)
        table = np.array([
            [np.sum((b_ref == 1) & (b_tgt == 1)), np.sum((b_ref == 1) & (b_tgt == 0))],
            [np.sum((b_ref == 0) & (b_tgt == 1)), np.sum((b_ref == 0) & (b_tgt == 0))],
        ])
        _, p_p = stats.fisher_exact(table, alternative='two-sided')

        # 3. Differential Spatial Pattern (Moran's I analytical z-test on SDM)
        D_i = x_tgt - x_ref  # signed spatial discrepancy map
        I_obs, p_s = _morans_I(D_i, W)

        records.append({
            'gene': gene_names[j],
            'abundance_pval': p_a,
            'abundance_cohens_d': cohens_d,
            'prevalence_pval': p_p,
            'spatial_pval': p_s,
            'spatial_morans_I': I_obs,
        })

    df = pd.DataFrame(records).set_index('gene')

    # Multiple testing correction per test type
    # Abundance: Bonferroni + Cohen's d effect-size gate
    pvals_a = df['abundance_pval'].values
    reject_a, padj_a, _, _ = multipletests(pvals_a, alpha=fdr_threshold, method='bonferroni')
    df['abundance_fdr'] = padj_a
    # Require both statistical significance AND meaningful effect size
    df['is_diff_abundance'] = reject_a & (np.abs(df['abundance_cohens_d'].values) >= abundance_effect_size)

    # Prevalence: Bonferroni
    pvals_p = df['prevalence_pval'].values
    reject_p, padj_p, _, _ = multipletests(pvals_p, alpha=fdr_threshold, method='bonferroni')
    df['prevalence_fdr'] = padj_p
    df['is_diff_prevalence'] = reject_p

    # Spatial: FDR-BH on analytical p-values + Moran's I effect-size gate
    pvals_s = df['spatial_pval'].values
    reject_s, padj_s, _, _ = multipletests(pvals_s, alpha=fdr_threshold, method='fdr_bh')
    df['spatial_fdr'] = padj_s
    # Require both statistical significance AND meaningful spatial autocorrelation
    df['is_diff_spatial'] = reject_s & (df['spatial_morans_I'].values >= spatial_I_threshold)

    # Build human-readable alteration_type column
    def _label(row):
        parts = []
        if row['is_diff_abundance']:
            parts.append('Abundance')
        if row['is_diff_prevalence']:
            parts.append('Prevalence')
        if row['is_diff_spatial']:
            parts.append('Spatial')
        return ' + '.join(parts) if parts else 'None'

    df['alteration_type'] = df.apply(_label, axis=1)

    return df
