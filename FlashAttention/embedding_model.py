import torch.nn as nn
import math
import torch

class PositionalEncoding(nn.Module):
    """Positional encoding."""
    
    def __init__(self, 
                 d_model: int, 
                 max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        
        # Ensure d_model is even for sin/cos pairing
        if d_model % 2 != 0:
            raise ValueError(f"d_model ({d_model}) must be even for sin/cos encoding")
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sin/cos calculation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even dimensions (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd dimensions (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, L, D]
        Returns:
            x + positional encoding: [B, L, D]
        """
        seq_len = x.size(1)
        
        # Check if sequence is too long
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_len ({self.pe.size(1)})")
        
        # Add positional encoding and apply dropout
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
    

class LearnablePositionalEncoding(nn.Module):
    """Alternative: Learnable positional embeddings."""
    
    def __init__(self, 
                 d_model: int, 
                 max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        # Learnable position embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Initialize with small random values
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, L, D]
        Returns:
            x + positional encoding: [B, L, D]
        """
        B, L, D = x.shape
        
        # Create position indices
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        
        # Add learnable position embeddings
        pos_emb = self.pos_embedding(positions)  # [B, L, D]
        x = x + pos_emb
        return self.dropout(x)
