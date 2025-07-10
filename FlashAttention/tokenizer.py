

class SimpleTokenizer:
    """Simple Tokenizer"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_token = 0
        self.unk_token = 1
        self.bos_token = 2  # Beginning of sequence
        self.eos_token = 3  # End of sequence
        
        # Reserve first 4 tokens for special tokens
        self.special_tokens = 4
        self.vocab_offset = self.special_tokens
        
    def encode(self, 
               text: str, 
               add_special_tokens: bool = True) -> list[int]:
        tokens = []
        
        # Add beginning of sequence token
        if add_special_tokens:
            tokens.append(self.bos_token)
        
        for char in text.lower():
            # Better character mapping
            if char.isalnum() or char in ' .,!?':  # Keep common punctuation
                # Map to available vocab space
                token_id = (ord(char) % (self.vocab_size - self.special_tokens)) + self.vocab_offset
            else:
                # Unknown character
                token_id = self.unk_token
            tokens.append(token_id)
        
        # Add end of sequence token
        if add_special_tokens:
            tokens.append(self.eos_token)
            
        return tokens
    
    def decode(self, 
               token_ids: list[int], 
               skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id < self.special_tokens:
                continue  # Skip special tokens
            elif token_id >= self.vocab_offset:
                # Reverse the encoding
                char_code = (token_id - self.vocab_offset) % 94 + 32  # Printable ASCII
                if 32 <= char_code <= 126:  # Valid printable range
                    chars.append(chr(char_code))
            # Skip invalid tokens
        return ''.join(chars)
    
    def pad_sequence(self, token_ids: list[int], max_length: int) -> list[int]:
        """Pad sequence to max_length."""
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        else:
            return token_ids + [self.pad_token] * (max_length - len(token_ids))