import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .sinkhorn import Sinkhorn



def pairwise_js_divergence_matrix(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
        p (torch.Tensor):  (K, N)
    """
    p1 = p.unsqueeze(1)
    p2 = p.unsqueeze(0)

    M = 0.5 * (p1 + p2)

    kl_p1_m = p1 * ((p1 + eps).log() - (M + eps).log())
    kl_p2_m = p2 * ((M + eps).log() - (p2 + eps).log())

    kl_div1 = kl_p1_m.sum(dim=-1)
    kl_div2 = kl_p2_m.sum(dim=-1)
    js_div_matrix = 0.5 * (kl_div1 + kl_div2)

    return js_div_matrix


def update_pi_gw_frank_wolfe(D_A, D_B, u, v, pi_old, device,
                             # --- Hyperparameters ---
                             gw_max_iter: int,
                             inner_sink_iter: int,
                             gw_epsilon: float):
    """
    Updates the transport plan pi using the Frank-Wolfe algorithm for Gromov-Wasserstein.
    """
    D_A_sq, D_B_sq = D_A**2, D_B**2
    pi = pi_old.clone() # Start with the old plan

    for _ in range(gw_max_iter):
        # 1. Calculate cost matrix (proportional to the gradient of the GW objective)
        cost_M = (D_A_sq @ u.unsqueeze(1) @ torch.ones(1, D_B.shape[0]).to(device) +
                  torch.ones(D_A.shape[0], 1).to(device) @ v.unsqueeze(0) @ D_B_sq.T -
                  2 * D_A @ pi @ D_B.T)
        
        if torch.isnan(cost_M).any() or torch.isinf(cost_M).any():
            print(f"Warning: cost_M in GW contains NaN/Inf. Stopping update and returning previous pi.")

        adaptive_epsilon = gw_epsilon
        
        pi_s = Sinkhorn.apply(cost_M, u, v, inner_sink_iter, adaptive_epsilon)

        pi_diff = pi_s - pi

        term_a = -2 * (D_A @ pi_diff @ D_B.T * pi_diff).sum()
        term_b = (cost_M * pi_diff).sum()

        gamma = -term_b / (term_a + 1e-8)
        gamma = torch.clamp(gamma, 0., 1.)
        
        pi = (1 - gamma) * pi + gamma * pi_s
        
    # return pi.detach()
    return pi


def OT_EM(s1, s2, anchor_mask, bsz, gene_size, codebook1, codebook2, pi, q_g_init, p0_init, kappa, device):

    # ------------------------------ Hyperparameters ------------------------------
    num_genes = s1.shape[0]
    K_A, K_B = s1.shape[1], s2.shape[1]
    # pi = torch.ones(K_A, K_B).to(device) / (K_A * K_B)
    
    # p0_beta_alpha, p0_beta_beta = p0_init * bsz, (1 - p0_init) * bsz
    p0_beta_alpha, p0_beta_beta = 200, 1000
    p0 = p0_init

    q_g = q_g_init.detach().to(device)

    beta = 0.01

    T, T_rho = 1, 1
    
    lambda_blend = 0.25

    gw_max_iter, inner_sink_iter, gw_epsilon = 10, 50, 1e-1

    epsilon = 1e-6

    s1_dist = s1
    s2_dist = s2
    gamma = 1.0
    # s1_prob = torch.exp(gamma * s1_dist)
    # s2_prob = torch.exp(gamma * s2_dist)
    # s1_prob = F.softmax(s1_dist, dim=1)
    # s2_prob = F.softmax(s2_dist, dim=1)

    # ------------------------------ Anchor-guided OT ------------------------------
    q_g_anchors = q_g[anchor_mask]
    s1_anchors = s1[anchor_mask] 
    s2_anchors = s2[anchor_mask]

    # p1_anchors_new = q_g_anchors.unsqueeze(1) * s1_anchors
    # p2_anchors_new = q_g_anchors.unsqueeze(1) * s2_anchors
    p1_anchors_new = s1_anchors # [A, K]
    p2_anchors_new = s2_anchors
    u_new = p1_anchors_new.sum(dim=0) / (p1_anchors_new.sum() + epsilon)
    v_new = p2_anchors_new.sum(dim=0) / (p2_anchors_new.sum() + epsilon)
    # u_new = torch.ones(K_A).to(device) / K_A
    # v_new = torch.ones(K_B).to(device) / K_B

    D_geom_A = torch.cdist(codebook1, codebook1)
    D_geom_B = torch.cdist(codebook2, codebook2)
    D_geom_A = D_geom_A / (D_geom_A.max() + epsilon)
    D_geom_B = D_geom_B / (D_geom_B.max() + epsilon)

    # print(f"DEBUG: D_geom_A max = {D_geom_A.max().item()}, D_geom_B max = {D_geom_B.max().item()}")
    
    p_j1 = p1_anchors_new.T # [K, A]
    p_j2 = p2_anchors_new.T
    tilde_p1 = p_j1 / (p_j1.sum(dim=1, keepdim=True) + epsilon)
    tilde_p2 = p_j2 / (p_j2.sum(dim=1, keepdim=True) + epsilon)
    tilde_p1 = torch.clamp(tilde_p1, min=0.0)
    tilde_p2 = torch.clamp(tilde_p2, min=0.0)
    D_anchor_A = pairwise_js_divergence_matrix(tilde_p1)
    D_anchor_B = pairwise_js_divergence_matrix(tilde_p2)
    D_anchor_A = D_anchor_A / (D_anchor_A.max() + epsilon)
    D_anchor_B = D_anchor_B / (D_anchor_B.max() + epsilon)
    
    D_A = D_anchor_A + lambda_blend * D_geom_A
    D_B = D_anchor_B + lambda_blend * D_geom_B

    # print(D_anchor_A.max(), D_geom_A.max())
    # print(D_anchor_B.max(), D_geom_B.max())

    pi = update_pi_gw_frank_wolfe(
        D_A, D_B, u_new, v_new, pi.detach(), device,
        gw_max_iter, inner_sink_iter, gw_epsilon
    )

    # ------------------------------ E-step ------------------------------
    pi_row_norm = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)  
    pi_col_norm = pi / (pi.sum(dim=0, keepdim=True) + 1e-8)

    hat_s2 = torch.matmul(s1, pi_row_norm) 
    hat_s1 = torch.matmul(s2, pi_col_norm.T)

    # p2 = torch.sum(pi, dim=0) + epsilon
    # hat_s2 = torch.matmul(s1_prob, pi / p2.unsqueeze(0))
    # # hat_s2 = (1.0 / gamma) * torch.log(hat_s2 + 1e-6)

    # p1 = torch.sum(pi, dim=1) + epsilon
    # hat_s1 = torch.matmul(s2_prob, (pi / p1.unsqueeze(1)).transpose(0, 1))
    # # hat_s1 = (1.0 / gamma) * torch.log(hat_s1 + 1e-6)

    # dist_s2 = 1.0 - F.cosine_similarity(hat_s2, s2_dist, dim=1)
    # dist_s1 = 1.0 - F.cosine_similarity(hat_s1, s1_dist, dim=1)
    # d_g = 0.5 * dist_s2 + 0.5 * dist_s1
    # d_g = 0.5 * (kl_divergence(hat_s2, s2_prob) + kl_divergence(hat_s1, s1_prob))
    d_g = 0.5 * F.mse_loss(hat_s2, s2_dist, reduction='none').mean(dim=1) + 0.5 * F.mse_loss(hat_s1, s1_dist, reduction='none').mean(dim=1)

    if kappa is None:
        kappa = torch.exp(-d_g.detach() / T).mean()
    else:
        non_anchor_exp = torch.exp(-d_g.detach()[~anchor_mask] / T)
        if non_anchor_exp.numel() > 0:
            non_anchor_mean = non_anchor_exp.mean()
        else:
            non_anchor_mean = 0.0 
        kappa = (1 - beta) * kappa.detach() + beta * non_anchor_mean

    p0_exp_term = p0 * torch.exp(-d_g.detach() / T)
    q_g = p0_exp_term / (p0_exp_term + (1 - p0) * kappa + 1e-8)
    q_g = torch.clamp(q_g, 0.0, 1.0)

    T = T * T_rho
    q_g = q_g.detach()
    # p0
    old_S = p0 * (p0_beta_alpha + p0_beta_beta - 2 + gene_size) - (p0_beta_alpha - 1)
    S = (1 - beta) * old_S + beta * (gene_size / bsz) * q_g[:bsz].sum() 
    p0 = (p0_beta_alpha - 1 + S) / (p0_beta_alpha + p0_beta_beta - 2 + gene_size)

    # ------------------------------ M-step ------------------------------
    loss = (q_g[:bsz] * d_g[:bsz]).mean()

    return loss, q_g, d_g, p0, kappa, pi


def get_final_transport_plan(
    q_g: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    codebook1: torch.Tensor,
    codebook2: torch.Tensor,
    anchor_mask: torch.Tensor,
    device: str,
    pi_init: torch.Tensor = None,
    # --- Hyperparameters ---
    lambda_blend: float = 0.2,
    gw_max_iter: int = 10,
    inner_sink_iter: int = 10,
    gw_epsilon: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the final Gromov-Wasserstein transport plan (pi) after the EM algorithm has converged.

    This function uses the final converged gene probabilities (q_g) to construct the
    final cost structure and solve for the optimal transport plan.

    Args:
        q_g (torch.Tensor): The converged gene probabilities, shape (num_genes,).
        s1 (torch.Tensor): Topic distributions for the first space, shape (num_genes, K_A).
        s2 (torch.Tensor): Topic distributions for the second space, shape (num_genes, K_B).
        codebook1 (torch.Tensor): Geometric positions of topics in the first space, shape (K_A, D).
        codebook2 (torch.Tensor): Geometric positions of topics in the second space, shape (K_B, D).
        anchor_mask (torch.Tensor): Boolean mask identifying anchor genes.
        device (str): The device to run computations on (e.g., 'cuda' or 'cpu').
        lambda_blend (float): Hyperparameter to blend geometric and feature-based distances.
        gw_max_iter (int): Number of iterations for the Gromov-Wasserstein solver.
        inner_sink_iter (int): Number of Sinkhorn iterations within the GW loop.
        gw_epsilon (float): Entropy regularization parameter for GW.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - pi (torch.Tensor): The final transport plan, shape (K_A, K_B).
            - u (torch.Tensor): The final marginal distribution for space 1, shape (K_A,).
            - v (torch.Tensor): The final marginal distribution for space 2, shape (K_B,).
    """
    # --- 1. Select anchor genes and compute marginals u and v ---
    K_A, K_B = s1.shape[1], s2.shape[1]
    s1_dist = s1
    s2_dist = s2
    # gamma = 1.0
    # s1_prob = torch.exp(gamma * s1_dist)
    # s2_prob = torch.exp(gamma * s2_dist)
    # s1_prob = F.softmax(s1_dist, dim=1)
    # s2_prob = F.softmax(s2_dist, dim=1)

    q_g_anchors = q_g[anchor_mask]
    s1_anchors = s1[anchor_mask]
    s2_anchors = s2[anchor_mask]

    # p1_anchors = q_g_anchors.unsqueeze(1) * s1_anchors
    # p2_anchors = q_g_anchors.unsqueeze(1) * s2_anchors
    p1_anchors = s1_anchors
    p2_anchors = s2_anchors
    
    # Normalize to get probability distributions u and v
    u = p1_anchors.sum(dim=0) / (p1_anchors.sum() + 1e-8)
    v = p2_anchors.sum(dim=0) / (p2_anchors.sum() + 1e-8)
    # u = torch.ones(K_A).to(device) / K_A
    # v = torch.ones(K_B).to(device) / K_B

    # --- 2. Construct the blended distance matrices D_A and D_B ---
    # Geometric distance
    D_geom_A = torch.cdist(codebook1, codebook1)
    D_geom_B = torch.cdist(codebook2, codebook2)
    D_geom_A = D_geom_A / (D_geom_A.max() + 1e-8)
    D_geom_B = D_geom_B / (D_geom_B.max() + 1e-8)
    
    # Feature-based distance (JS-divergence)
    p_j1 = p1_anchors.T
    p_j2 = p2_anchors.T
    tilde_p1 = p_j1 / (p_j1.sum(dim=1, keepdim=True) + 1e-8)
    tilde_p2 = p_j2 / (p_j2.sum(dim=1, keepdim=True) + 1e-8)
    # tilde_p1 = F.softmax(p_j1, dim=1)
    # tilde_p2 = F.softmax(p_j2, dim=1)
    D_anchor_A = pairwise_js_divergence_matrix(tilde_p1)
    D_anchor_B = pairwise_js_divergence_matrix(tilde_p2)
    D_anchor_A = D_anchor_A / (D_anchor_A.max() + 1e-8)
    D_anchor_B = D_anchor_B / (D_anchor_B.max() + 1e-8)
    
    # Blend the two distances
    D_A = (1 - lambda_blend) * D_anchor_A + lambda_blend * D_geom_A
    D_B = (1 - lambda_blend) * D_anchor_B + lambda_blend * D_geom_B

    # --- 3. Solve for the transport plan pi ---
    # Initialize a starting pi for the iterative solver
    if pi_init is None:
        K_A, K_B = s1.shape[1], s2.shape[1]
        pi_init = torch.ones(K_A, K_B, device=device) / (K_A * K_B)
    
    pi = update_pi_gw_frank_wolfe(
        D_A, D_B, u, v, pi_init, device,
        gw_max_iter, inner_sink_iter, gw_epsilon
    )
    
    return pi, u, v


def transform_distributions(
    s1: torch.Tensor,
    s2: torch.Tensor,
    pi: torch.Tensor,
    u: torch.Tensor = None,
    v: torch.Tensor = None,
    eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transforms distributions between space 1 and space 2 using the transport plan.

    Args:
        s1 (torch.Tensor): The full set of distributions in space 1, shape (num_genes, K_A).
        s2 (torch.Tensor): The full set of distributions in space 2, shape (num_genes, K_B).
        pi (torch.Tensor): The transport plan from space 1 to 2, shape (K_A, K_B).
        u (torch.Tensor): The marginal distribution for space 1, shape (K_A,).
        v (torch.Tensor): The marginal distribution for space 2, shape (K_B,).
        eps (float): A small epsilon for numerical stability.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - s2_hat (torch.Tensor): s1 transformed into space 2, shape (num_genes, K_B).
            - s1_hat (torch.Tensor): s2 transformed into space 1, shape (num_genes, K_A).
    """
    # --- Forward transformation: s1 -> s2 ---
    # Normalize s1 based on the marginal u. This step is crucial.
    # It corresponds to the `s1_scaled` operation in your e_step.
    # s1_scaled = s1 / (u.unsqueeze(0) + eps)
    
    # # Apply the transport plan to map from space A to space B
    # hat_s2 = s1_scaled @ pi

    # # --- Reverse transformation: s2 -> s1 ---
    # # The reverse transport plan is the transpose of pi
    # pi_reverse = pi.T
    
    # # Normalize s2 based on the marginal v
    # s2_scaled = s2 / (v.unsqueeze(0) + eps)
    
    # # Apply the reverse transport plan to map from space B to space A
    # hat_s1 = s2_scaled @ pi_reverse
    pi_row_norm = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)  
    pi_col_norm = pi / (pi.sum(dim=0, keepdim=True) + 1e-8)

    hat_s2 = torch.matmul(s1, pi_row_norm) 
    hat_s1 = torch.matmul(s2, pi_col_norm.T)

    return hat_s2, hat_s1




