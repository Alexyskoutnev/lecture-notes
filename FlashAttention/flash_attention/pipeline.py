import torch

from tokenizer import SimpleTokenizer
from config import MODEL_PARAMS
from flash_attention.flash_attention_transformer import FlashAttentionTransformer

def main():
    """Test the complete Flash Attention transformer pipeline."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    VOCAB_SIZE = MODEL_PARAMS.get('VOCAB', 1000)
    D_MODEL = MODEL_PARAMS.get('D_MODEL', 256)
    N_HEADS = MODEL_PARAMS.get('N_HEADS', 8)
    N_LAYERS = 1  # Multiple layers
    SEQ_LEN = MODEL_PARAMS.get('SEQ_LEN', 64)

    
    # Create complete Flash Attention transformer
    transformer = FlashAttentionTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.1,
        debug=False  # Set to True to see attention visualizations
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"Model device: {next(transformer.parameters()).device}")
    
    # Test data
    tokenizer = SimpleTokenizer(VOCAB_SIZE)
    text = "Hello world, welcome to this lecture series from Nice Intelligence"
    print(f"\nInput text: '{text}'")
    
    # Tokenize and prepare
    token_ids = tokenizer.encode(text)
    print(f"Token IDs: {token_ids}")
    
    # Pad to sequence length
    if len(token_ids) < SEQ_LEN:
        token_ids.extend([tokenizer.pad_token] * (SEQ_LEN - len(token_ids)))
    else:
        token_ids = token_ids[:SEQ_LEN]
    
    tokens = torch.tensor([token_ids], device=device)  # [1, SEQ_LEN]
    
    
    # Forward pass
    with torch.no_grad():
        context = transformer(tokens, causal=False)
        print(f"Context shape: {context.shape}")
        print(f"Context dtype: {context.dtype}")
        print(f"Sample context: {context[0, 0, :6]}")

if __name__ == "__main__":
    main()