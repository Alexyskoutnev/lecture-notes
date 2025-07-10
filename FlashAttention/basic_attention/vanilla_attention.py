import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor, 
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None= None
) -> Tensor:
    """
    implementation of PyTorch's F.scaled_dot_product_attention.
    
    Args:
        query: [B, H, L, D] - Query tensor
        key: [B, H, L, D] - Key tensor  
        value: [B, H, L, D] - Value tensor
        attn_mask: [L, L] or [B, H, L, L] - Attention mask (True = mask out)
        dropout_p: Dropout probability for attention weights
        is_causal: Whether to apply causal (lower triangular) mask
        scale: Scale factor (defaults to 1/sqrt(d_k))
        
    Returns:
        output: [B, H, L, D] - Attention output
    """
    B, H, L, D = query.shape
    
    # 1. Compute scale factor
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # 2. Compute attention scores: Q @ K^T
    # query: [B, H, L, D], key: [B, H, L, D]
    # scores: [B, H, L, L] where scores[b,h,i,j] = similarity between query_i and key_j
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # 3. Apply causal mask if requested
    if is_causal:
        # Create lower triangular mask (upper triangle = True = mask out future tokens)
        causal_mask = torch.triu(
            torch.ones(L, L, device=query.device, dtype=torch.bool),
            diagonal=1
        )
        scores.masked_fill_(causal_mask, float('-inf'))
    
    # 4. Apply custom attention mask if provided
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # Boolean mask: True means mask out (set to -inf)
            scores.masked_fill_(attn_mask, float('-inf'))
        else:
            # Additive mask: add the mask values
            scores = scores + attn_mask
    
    # 5. Apply softmax to get attention probabilities
    attn_weights = F.softmax(scores, dim=-1)  # [B, H, L, L]
    
    # 6. Apply dropout to attention weights (if training)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    
    # 7. Apply attention to values: P @ V
    # attn_weights: [B, H, L, L], value: [B, H, L, D]
    # output: [B, H, L, D]
    output = torch.matmul(attn_weights, value)
    
    return output


class VanillaMultiHeadAttention(nn.Module):
    """Vanilla multi-head attention."""
    
    def __init__(self,
                d_model: int, 
                n_heads: int, 
                dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Separate projections for Q, K, V (more common than combined)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False) 
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        B, L, _ = x.shape
        
        # Generate Q, K, V
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)  # [B, L, D] 
        v = self.v_proj(x)  # [B, L, D]
        
        # Reshape for multi-head: [B, L, H, Dh] -> [B, H, L, Dh]
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention computation
        attn_mask = None
        if causal:
            attn_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        
        # Vanilla Attention Compute
        out = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Reshape back: [B, H, L, Dh] -> [B, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        return self.out_proj(out)