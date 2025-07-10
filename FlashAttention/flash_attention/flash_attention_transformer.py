
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from flash_attention.flash_attention_fn import flash_attention_fwd
from tokenizer import SimpleTokenizer
from embedding_model import PositionalEncoding
from config import MODEL_PARAMS
from ff_model import FeedForward

class FlashMHAttention(nn.Module):
    """Multi-head self-attention layer using Flash Attention implementation."""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int,
                 dropout: float = 0.1,
                 debug: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.tile_block_q = 16  # Tiling given hardware constraints H100, A100 etc.
        self.tile_block_k = 16  # Tiling given hardware constraints H100, A100 etc.
        self.debug_mode = debug
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        # [B, L, 3*D] -> [B, L, H, Dh]
        B, L, _ = x.shape
        x = x.view(B, L, self.n_heads, self.d_head)
        return x

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x)  # [B, L, 3*D]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q, k, v = map(self._split_heads, (q, k, v))
        
        # Flash Attention kernel
        ctx = flash_attention_fwd(q, k, v,
                                causal=causal,
                                block_q=self.tile_block_q,
                                block_k=self.tile_block_k,
                                debug=self.debug_mode)  # [B, L, H, Dh]
        
        # Merge heads back
        ctx = ctx.contiguous().view(B, L, -1)  # [B, L, D]
        
        # Apply dropout to attention output
        ctx = self.dropout(ctx)
        
        return self.out(ctx)  # Final projection

class FlashTransformerBlock(nn.Module):
    """Complete transformer block with Flash Attention, feed-forward, and residual connections."""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int = None, 
                 dropout: float = 0.1,
                 debug: bool = False):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model  # Standard ratio
            
        self.attention = FlashMHAttention(d_model, n_heads, dropout, debug)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer norms (Pre-LN is more stable than Post-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
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
    

class FlashAttentionTransformer(nn.Module):
    """Complete Flash Attention Transformer."""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int, 
                 n_heads: int,
                 n_layers: int = 1,
                 d_ff: int = None,
                 max_seq_len: int = 1024,
                 dropout: float = 0.1,
                 debug: bool = False):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.d_model = d_model
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)
        
        # Embedding scaling
        self.embed_scale = math.sqrt(d_model)
        
        # Flash Attention transformer blocks
        self.blocks = nn.ModuleList([
            FlashTransformerBlock(d_model, n_heads, d_ff, dropout, debug)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, 
                token_ids: Tensor, 
                causal: bool = False) -> Tensor:
        """
        Args:
            token_ids: [B, L] - input token IDs
            causal: whether to use causal masking
        Returns:
            context vectors: [B, L, D]
        """
        # Token embedding with scaling
        x = self.token_embed(token_ids) * self.embed_scale  # [B, L, D]
        
        # Add positional encoding
        x = self.pos_encoding(x)  # [B, L, D]
        x = self.embed_dropout(x)
        
        # Pass through Flash Attention transformer blocks
        for block in self.blocks:
            x = block(x, causal=causal)
        
        # Final layer norm
        x = self.final_norm(x)
        
        return x

# ------------------------------ quick test ------------------------------
if __name__ == "__main__":
    """Test the simple transformer with sample text."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    VOCAB_SIZE = MODEL_PARAMS.get('VOCAB', 1000)
    D_MODEL = MODEL_PARAMS.get('D_MODEL', 128)
    N_HEADS = MODEL_PARAMS.get('N_HEADS', 8)
    SEQ_LEN = MODEL_PARAMS.get('SEQ_LEN', 64)
    
    model = FlashAttentionTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        debug=True
    ).to(device)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(VOCAB_SIZE)
    
    # Sample text
    text = "Hello world, welcome to this lecture series from Nice Intelligence"
    print(f"Input text: '{text}'")
    
    # Tokenize
    token_ids = tokenizer.encode(text)
    print(f"Token IDs: {token_ids}")
    
    # Pad to sequence length
    # SEQ_LENGTH is the context window for the model
    if len(token_ids) < SEQ_LEN:
        token_ids.extend([tokenizer.pad_token] * (SEQ_LEN - len(token_ids)))
    else:
        token_ids = token_ids[:SEQ_LEN]
    
    # Convert to tensor
    tokens = torch.tensor([token_ids], device=device)  # [1, SEQ_LEN]
    print(f"Input shape: {tokens.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(tokens)