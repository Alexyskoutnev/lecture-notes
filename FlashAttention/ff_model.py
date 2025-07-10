import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1, 
                 activation: str = "relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))