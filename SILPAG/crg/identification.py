"""
CRG (Condition-Responsive Gene) Identification.

Computes per-gene CRG scores based on the Frobenius norm discrepancy between
the view-transferred SGM and the observed target SGM, then tests significance
against a null derived from anchor genes.
"""

import numpy as np
import anndata as ad
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


def compute_crg_scores(
    result_ref: ad.AnnData,
    adata: ad.AnnData,
) -> np.ndarray:
    """
    Compute per-gene CRG score S_i = ||x_ref->tgt_i - x_tgt_i||_F^2.

    Parameters
    ----------
    result_ref : AnnData
        View-transferred reconstruction (from reference codebook via OT).
    adata : AnnData
        Original target AnnData.

    Returns
    -------
    scores : np.ndarray, shape (n_genes,)
    """
    X_ref = result_ref.X
    X_tgt = adata.X
    if sparse.issparse(X_ref):
        X_ref = X_ref.toarray()
    if sparse.issparse(X_tgt):
        X_tgt = X_tgt.toarray()

    diff = X_ref - X_tgt
    scores = np.sum(diff ** 2, axis=0)
    return scores


def identify_crgs(
    scores: np.ndarray,
    anchor_mask: np.ndarray,
    p_threshold: float = 0.05,
    null_model: str = "zscore",
    correction: str = "fdr_bh",
    top_k: int = 200,
    fc_threshold: float = 2.0,
) -> dict:
    """
    Identify CRGs by testing each gene's score against a null from anchors.

    Parameters
    ----------
    scores : np.ndarray, shape (n_genes,)
        CRG scores from compute_crg_scores.
    anchor_mask : np.ndarray, shape (n_genes,)
        Boolean or 0/1 array indicating anchor genes (expected stable).
    p_threshold : float
        Significance cutoff. Applied to adjusted p-values when correction
        is used, or to raw p-values when correction=None.
    null_model : str
        How to model the null distribution from anchor scores:
        - 'gamma'      : fit Gamma(k, theta) via MLE
        - 'zscore'     : Gaussian z-score on log-transformed scores
        - 'percentile' : empirical percentile rank against anchors
        - 'topk'       : simply select top_k genes by score (no statistical test)
        - 'fold_change': select genes whose score > fc_threshold × median(anchor scores)
    correction : str or None
        Multiple testing correction method (e.g. 'fdr_bh').
        Set to None to skip FDR correction and use raw p-values directly.
    top_k : int
        Number of top-scoring genes to select when null_model='topk'.
    fc_threshold : float
        Fold-change multiplier over median anchor score when null_model='fold_change'.

    Returns
    -------
    result : dict with keys:
        'scores', 'pvalues', 'pvalues_adj', 'is_crg', 'null_info'
    """
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    S_null = scores[anchor_mask]
    n_genes = len(scores)

    if null_model == "topk":
        # No statistical testing — just pick the top_k highest-scoring genes
        # Exclude anchor genes from selection
        non_anchor_mask = ~anchor_mask
        ranked_idx = np.argsort(scores)[::-1]
        selected = np.zeros(n_genes, dtype=bool)
        count = 0
        for idx in ranked_idx:
            if count >= top_k:
                break
            if non_anchor_mask[idx]:
                selected[idx] = True
                count += 1
        # Assign pseudo p-values based on rank for downstream compatibility
        rank = np.argsort(np.argsort(-scores)).astype(float)
        pvalues = rank / n_genes
        pvalues_adj = pvalues.copy()
        null_info = {"model": "topk", "top_k": top_k, "n_selected": int(selected.sum())}
        return {
            "scores": scores,
            "pvalues": pvalues,
            "pvalues_adj": pvalues_adj,
            "is_crg": selected,
            "null_info": null_info,
        }

    elif null_model == "fold_change":
        # Simple fold-change over anchor median — no FDR correction
        anchor_median = np.median(S_null)
        threshold = fc_threshold * anchor_median
        is_crg = (scores > threshold) & (~anchor_mask)
        # Pseudo p-values: score / max_score inverted
        pvalues = 1.0 - (scores / (scores.max() + 1e-12))
        pvalues_adj = pvalues.copy()
        null_info = {
            "model": "fold_change",
            "anchor_median": float(anchor_median),
            "threshold": float(threshold),
            "fc_threshold": fc_threshold,
        }
        return {
            "scores": scores,
            "pvalues": pvalues,
            "pvalues_adj": pvalues_adj,
            "is_crg": is_crg,
            "null_info": null_info,
        }

    elif null_model == "gamma":
        k, loc, theta = stats.gamma.fit(S_null, floc=0)
        pvalues = 1.0 - stats.gamma.cdf(scores, k, loc=loc, scale=theta)
        null_info = {"model": "gamma", "k": k, "loc": loc, "theta": theta}

    elif null_model == "zscore":
        # Log-transform to stabilise variance, then z-score
        log_null = np.log1p(S_null)
        mu, sigma = log_null.mean(), log_null.std(ddof=1)
        z = (np.log1p(scores) - mu) / (sigma + 1e-12)
        pvalues = 1.0 - stats.norm.cdf(z)
        null_info = {"model": "zscore", "mu": mu, "sigma": sigma}

    elif null_model == "percentile":
        # Empirical: fraction of anchors with score >= s_i
        pvalues = np.array([np.mean(S_null >= s) for s in scores])
        # Clamp minimum p-value to 1/(n_anchor+1) to avoid exact zeros
        pvalues = np.maximum(pvalues, 1.0 / (len(S_null) + 1))
        null_info = {"model": "percentile", "n_anchor": len(S_null)}

    else:
        raise ValueError(
            f"Unknown null_model '{null_model}'. "
            "Use 'gamma', 'zscore', 'percentile', 'topk', or 'fold_change'."
        )

    # Apply correction or use raw p-values
    if correction is not None:
        reject, pvalues_adj, _, _ = multipletests(
            pvalues, alpha=p_threshold, method=correction
        )
    else:
        pvalues_adj = pvalues.copy()
        reject = pvalues < p_threshold

    return {
        "scores": scores,
        "pvalues": pvalues,
        "pvalues_adj": pvalues_adj,
        "is_crg": reject,
        "null_info": null_info,
    }
