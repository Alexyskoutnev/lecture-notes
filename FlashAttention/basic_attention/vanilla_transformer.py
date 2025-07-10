import torch.nn as nn
import torch
import math

from basic_attention.vanilla_attention import VanillaMultiHeadAttention
from ff_model import FeedForward
from embedding_model import PositionalEncoding

class VanillaTransformerBlock(nn.Module):
    """Complete transformer block with residual connections and layer norms."""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int = None, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_ff = d_ff if d_ff else d_model * 4
        self.attention = VanillaMultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, self.d_ff, dropout)
        
        # Layer norms (Pre-LN is more stable than Post-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                causal: bool = False) -> torch.Tensor:
        # Convert Token Ids to embeddings
        # Pre-LN residual connection for attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, causal=causal)
        x = self.dropout(x) + residual
        
        # Pre-LN residual connection for feed-forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x) + residual
        
        return x
    
class VanillaTransformer(nn.Module):
        
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int = 1,
                 max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Token embedding layer (converts token IDs to dense vectors)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional encoding (adds position information)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 3. Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)
        
        # 4. Embedding scaling (standard practice)
        self.embed_scale = math.sqrt(d_model)
        
        # 5. Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            VanillaTransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # 6. Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                token_ids: torch.Tensor, 
                causal: bool = False) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] - integer token IDs
            causal: whether to use causal masking
        Returns:
            context: [B, L, D] - contextualized representations
        """
        # Step 1: Convert token IDs to embeddings
        x = self.token_embed(token_ids)  # [B, L, D]
        
        # Step 2: Scale embeddings (standard practice)
        x = x * self.embed_scale
        
        # Step 3: Add positional encoding
        x = self.pos_encoding(x)  # [B, L, D]
        
        # Step 4: Apply embedding dropout
        x = self.embed_dropout(x)
        
        # Step 5: Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal=causal)
        
        # Step 6: Final layer norm
        x = self.final_norm(x)
        
        return x