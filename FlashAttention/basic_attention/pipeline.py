import torch
import math
import torch.nn as nn
from config import MODEL_PARAMS
from basic_attention.vanilla_transformer import VanillaTransformerBlock
from tokenizer import SimpleTokenizer
from embedding_model import PositionalEncoding
from basic_attention.vanilla_transformer import VanillaTransformer


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    VOCAB_SIZE = MODEL_PARAMS.get('VOCAB', 1000)
    D_MODEL = MODEL_PARAMS.get('D_MODEL', 256)
    N_HEADS = MODEL_PARAMS.get('N_HEADS', 8)
    N_LAYERS = 1
    SEQ_LEN = MODEL_PARAMS.get('SEQ_LEN', 64)
    
    print("="*60)
    print("COMPLETE TRANSFORMER PIPELINE TEST")
    print("="*60)
    
    transformer = VanillaTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=SEQ_LEN,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Test data
    tokenizer = SimpleTokenizer(VOCAB_SIZE)
    text = "Hello world, welcome to this lecture series! -> Alexy"
    print(f"\nInput text: '{text}'")
    
    # Step 1: Tokenize
    token_ids = tokenizer.encode(text)
    print(f"Token IDs: {token_ids}")
    
    # Step 2: Pad sequence to fixed length
    if len(token_ids) < SEQ_LEN:
        token_ids.extend([tokenizer.pad_token] * (SEQ_LEN - len(token_ids)))
    else:
        token_ids = token_ids[:SEQ_LEN]
    
    # Step 3: Convert to tensor and add batch dimension
    tokens = torch.tensor([token_ids], device=device)  # [1, SEQ_LEN]
    
    with torch.no_grad():
        # Regular attention
        context = transformer(tokens, causal=False)
        print(f"Output shape: {context.shape}")
        
if __name__ == "__main__":
    main()