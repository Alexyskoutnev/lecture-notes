"""
flash_attention_pt_annotated.py
--------------------------------
Pure-PyTorch, training-friendly Flash-Attention *forward* kernel
that illustrates the two-pass streaming
algorithm from Dao et al. 2022.

Key ideas this code demonstrates
  • Tile the sequence dimension (L) so only Bq×Bk score blocks
    live in SRAM / registers at any time.
  • Pass-1  : compute row-max  m_i   and running denominator  l_i
  • Pass-2  : recompute scores, normalise, multiply by V, accumulate O
  • Causal masking only touches scores inside each tile.

The implementation keeps the math 100 % exact; speed mainly comes
from reduced HBM traffic.
"""

from __future__ import annotations
import math, torch
from torch import Tensor
from display_mtx import display_o_all_heads


# -------------------------------------------------------------------------
# Helper 1 — batched (Q_i · K_jᵀ) using torch.bmm
# -------------------------------------------------------------------------
def _qk_scores(Qi: Tensor, 
               Kj: Tensor, 
               scale: float) -> Tensor:
    """
    Computes the scaled dot-product scores for one *query-tile* Qi and one
    *key-tile* Kj *per attention head*.

    Args
    ----
    Qi : [Bq, H, Dh]   query slice  (Bq  rows,  H heads, Dh dim)
    Kj : [Bk, H, Dh]   key   slice  (Bk  rows)
    scale : 1/sqrt(Dh) pre-computed scaling factor

    Returns
    -------
    S : [Bq, Bk, H]    scores arranged so   S[q, k, h]  is the dot-product
    """
    # 1) Move the head axis into the *batch* position so each head becomes
    #    an independent matrix multiply in a batched GEMM call.
    #      Qi_bh : [H, Bq, Dh]
    Qi_bh = Qi.transpose(0, 1).contiguous()
    #      Kj_bh : [H, Bk, Dh]
    Kj_bh = Kj.transpose(0, 1).contiguous()

    # 2) For each head h:  [Bq, Dh] @ [Dh, Bk] → [Bq, Bk]
    S_bh = torch.bmm(Qi_bh, Kj_bh.transpose(1, 2))          # [H, Bq, Bk]

    # 3) Restore layout expected by the rest of the kernel: [Bq, Bk, H]
    #    and apply the scaling factor (√d_k in paper).
    return S_bh.permute(1, 2, 0).contiguous() * scale


# -------------------------------------------------------------------------
# Helper 2 — accumulate   P_ij @ V_j    using torch.bmm
# -------------------------------------------------------------------------
def _pv_accum(P: Tensor, 
              Vj: Tensor) -> Tensor:
    """
    Contracts  P (probabilities)  with  Vj  (values)  for the current
    (query-tile, key-tile) pair and returns the partial output.

    Args
    ----
    P  : [Bq, Bk, H]   attention probabilities for this tile
    Vj : [Bk,    H, Dh]
    Returns
    -------
    O_part : [Bq, H, Dh]   contribution to O from this tile pair
    """
    # Bring head to batch and align matmul inner dims
    P_bh  = P.permute(2, 0, 1).contiguous()          # [H, Bq, Bk]
    Vj_bh = Vj.transpose(0, 1).contiguous()          # [H, Bk, Dh]
    
    # Ensure both tensors have the same dtype
    P_bh = P_bh.to(Vj_bh.dtype)
    
    out_bh = torch.bmm(P_bh, Vj_bh)                  # [H, Bq, Dh]
    return out_bh.permute(1, 0, 2).contiguous()      # [Bq, H, Dh]


# -------------------------------------------------------------------------
# Main kernel — FlashAttention forward (two streaming passes)
# -------------------------------------------------------------------------
@torch.no_grad()
def flash_attention_fwd(
    q: Tensor,                       # [B, Lq, H, Dh]
    k: Tensor,                       # [B, Lk, H, Dh]
    v: Tensor,                       # [B, Lk, H, Dh]
    *,
    causal: bool = False,            # if True, causal mask  (decoder attn)
    block_q: int = 16,              # tile height   (tune to shared-mem size)
    block_k: int = 16,              # tile width    (ditto)
    debug: bool = False,
) -> Tensor:                         # returns [B, Lq, H, Dh]
    """
    Streaming Flash-Attention forward pass.

    • Works for self-attention (Lq == Lk) and cross-attention.
    • Assumes fp16/bf16/fp32 inputs already live on GPU.

    Complexity
    ----------
      FLOPs  : O(Lq·Lk·H·Dh)   (same as vanilla attention)
      HBM IO : O(L·H·Dh)       (each Q/K/V/O streamed once)
    """
    # -------------------------- sanity checks ---------------------------
    assert q.ndim == k.ndim == v.ndim == 4
    B, Lq, H, Dh = q.shape
    _, Lk, _, _  = k.shape
    scale = 1.0 / math.sqrt(Dh)                       # √d_k factor

    # ---------------------- buffers in global mem ----------------------
    o = torch.zeros_like(q)                           # final output O
    m = torch.full((B, Lq, H), -float('inf'), device=q.device, dtype=q.dtype)  # row-maxes
    l = torch.zeros((B, Lq, H), device=q.device, dtype=q.dtype)      # denominators

    # ---------------------- tile helper lambdas ------------------------
    def q_tile(b, tq): return q[b, tq:tq + block_q]   # [Bq, H, Dh]
    def k_tile(b, sk): return k[b, sk:sk + block_k]   # [Bk, H, Dh]
    def v_tile(b, sk): return v[b, sk:sk + block_k]

    # ======================== PASS 1 ===================================
    # Goal:  for every query row  i  collect
    #          m_i  =  max_k  score(i,k)
    #          l_i  = Σ_k  exp(score(i,k) - m_i)
    # These fit in registers/shared memory per tile.
    # ===================================================================
    for b in range(B):
        for tq in range(0, Lq): # iterate over each Q segment
            Qi   = q_tile(b, tq)                      # load one Q-tile
            m_i  = m[b, tq:tq + Qi.shape[0]]          # view into global
            l_i  = l[b, tq:tq + Qi.shape[0]]

            for sk in range(0, Lk, block_k):          # stream over K/V tiles
                if causal and tq >= sk + block_k:     # tile fully to the left
                    continue                          # → scores masked out

                Kj = k_tile(b, sk)
                S  = _qk_scores(Qi, Kj, scale)        # [Bq, Bk, H]

                # Apply row-wise causal mask to *partial* tile if needed
                if causal and tq < sk + block_k:      # overlaps future tokens
                    q_idx = torch.arange(tq, tq + Qi.shape[0], device=q.device)
                    k_idx = torch.arange(sk, sk + Kj.shape[0], device=q.device)
                    S[q_idx[:, None] < k_idx[None, :]] = -float('inf')
                # Update running max / denom  (online soft-max trick)
                m_new = torch.maximum(m_i, S.amax(dim=1))        # new row-max
                exp_shift = torch.exp(m_i - m_new)               # scale old L
                l_i.mul_(exp_shift).add_(                       # l_i = old*α + new
                    torch.exp(S - m_new[:, None]).sum(dim=1)
                )
                m_i.copy_(m_new)                                # write back

    # ======================== PASS 2 ===================================
    # Goal:  recompute scores, convert to prob P_ij, accumulate O_i.
    #         O_i = Σ_j  P_ij  ·  V_j
    # ===================================================================
    for b in range(B):
        for idx, tq in enumerate(range(0, Lq, block_q)): # Iterative over each query segment
            Qi  = q_tile(b, tq)                                 # [Bq, H, Dh]
            m_i = m[b, tq:tq + Qi.shape[0]]                     # [Bq, H]
            l_i = l[b, tq:tq + Qi.shape[0]]                     # [Bq, H]
            Oi  = torch.zeros_like(Qi)                          # local output

            for sk in range(0, Lk, block_k):
                if causal and tq >= sk + block_k:
                    continue
                Kj = k_tile(b, sk)
                Vj = v_tile(b, sk)
                S  = _qk_scores(Qi, Kj, scale)

                if causal and tq < sk + block_k:                # partial mask
                    q_idx = torch.arange(tq, tq + Qi.shape[0], device=q.device)
                    k_idx = torch.arange(sk, sk + Kj.shape[0], device=q.device)
                    S[q_idx[:, None] < k_idx[None, :]] = -float('inf')

                # Normalise:  P_ij = exp(score - m_i) / l_i
                P = torch.exp(S - m_i[:, None]) / l_i[:, None]  # [Bq, Bk, H]
                # Accumulate tile contribution to O_i
                Oi += _pv_accum(P, Vj)                          # [Bq, H, Dh]
            # Copy finished tile back to global output tensor
            o[b, tq:tq + Qi.shape[0]].copy_(Oi)
            if debug:
                display_o_all_heads(o=o, title=f"Flash Attention Mtx [{idx}]")

    return o