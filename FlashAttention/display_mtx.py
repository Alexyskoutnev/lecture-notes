import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def display_o_all_heads(o, 
                        batch_idx=0, 
                        title="Flash Attention Output - All Heads"):
    """
    Display the output matrix o showing all attention heads at once.
    
    Args:
        o: Output tensor [B, L, H, D]
        batch_idx: Which batch to display
        title: Title for the plot
    """
    # Convert to numpy
    if isinstance(o, torch.Tensor):
        o_np = o.detach().cpu().numpy()
    else:
        o_np = np.array(o)
    
    B, L, H, D = o_np.shape
    
    # Reshape to show all heads: [L, H*D]
    # This concatenates all heads horizontally
    matrix = o_np[batch_idx].reshape(L, H * D)  # [L, H*D]

    plt.figure(figsize=(max(12, H * 2), 8))
    
    # Create the heatmap
    ax = sns.heatmap(matrix, 
                     cmap='RdBu_r', 
                     center=0,
                     cbar=True,
                     xticklabels=False,
                     yticklabels=True)
    
    # Add vertical lines to separate heads
    for h in range(1, H):
        ax.axvline(x=h * D, color='white', linewidth=2)
    
    # Add head labels at the top
    for h in range(H):
        ax.text(h * D + D/2, -0.5, f'Head {h}', 
                ha='center', va='top', fontweight='bold')
        
    plt.savefig(f"{title.replace(' ', '_')}_batch_{batch_idx}.png", dpi=150, bbox_inches='tight')
