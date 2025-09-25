"""
Step-by-Step Learning Guide: Building NMT from Scratch
This file explains each component in detail for educational purposes
"""

import torch
import torch.nn as nn

# =============================================================================
# STEP 1: Understanding Embeddings
# =============================================================================
def create_embedding_example():
    """
    Learn how embeddings work
    """
    vocab_size = 1000  # Total unique words
    embed_dim = 256    # Each word becomes 256-dimensional vector
    
    # Create embedding layer
    embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    
    # Example: Convert word IDs to vectors
    word_ids = torch.tensor([1, 15, 234, 0])  # 0 is padding
    embedded = embedding(word_ids)
    print(f"Word IDs: {word_ids}")
    print(f"Embedded shape: {embedded.shape}")  # [4, 256]
    
    return embedding

# =============================================================================
# STEP 2: Understanding LSTM
# =============================================================================
def create_lstm_example():
    """
    Learn how LSTM processes sequences
    """
    input_size = 256   # Embedding dimension
    hidden_size = 512  # LSTM hidden state size
    num_layers = 2     # Stack 2 LSTM layers
    
    # Create LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                   batch_first=True, bidirectional=True)
    
    # Example input: batch of 3 sentences, max length 10 words
    batch_size, seq_len = 3, 10
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output, (hidden, cell) = lstm(x)
    
    print(f"Input shape: {x.shape}")           # [3, 10, 256]
    print(f"Output shape: {output.shape}")     # [3, 10, 1024] (512*2 for bidirectional)
    print(f"Hidden shape: {hidden.shape}")     # [4, 3, 512] (2 layers * 2 directions)
    
    return lstm

# =============================================================================
# STEP 3: Understanding Attention
# =============================================================================
def create_attention_example():
    """
    Learn how attention mechanism works
    """
    class SimpleAttention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1)
        
        def forward(self, decoder_hidden, encoder_outputs):
            # decoder_hidden: [batch, hidden_dim]
            # encoder_outputs: [batch, seq_len, hidden_dim]
            
            batch_size, seq_len, hidden_dim = encoder_outputs.shape
            
            # Repeat decoder hidden for each encoder position
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            
            # Concatenate and calculate attention scores
            energy = torch.tanh(self.attention(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
            attention_scores = self.v(energy).squeeze(2)
            
            # Apply softmax to get attention weights
            attention_weights = torch.softmax(attention_scores, dim=1)
            
            # Calculate context vector
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            
            return context.squeeze(1), attention_weights
    
    # Example usage
    hidden_dim = 512
    attention = SimpleAttention(hidden_dim)
    
    batch_size, seq_len = 2, 8
    decoder_hidden = torch.randn(batch_size, hidden_dim)
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim)
    
    context, weights = attention(decoder_hidden, encoder_outputs)
    
    print(f"Decoder hidden: {decoder_hidden.shape}")    # [2, 512]
    print(f"Encoder outputs: {encoder_outputs.shape}")  # [2, 8, 512]
    print(f"Context vector: {context.shape}")           # [2, 512]
    print(f"Attention weights: {weights.shape}")        # [2, 8]
    print(f"Attention weights sum: {weights.sum(dim=1)}")  # Should be [1, 1]
    
    return attention

# =============================================================================
# STEP 4: Understanding BPE Tokenization
# =============================================================================
def bpe_learning_example():
    """
    Learn how BPE works step by step
    """
    # Start with a small corpus
    corpus = ["محبت میں", "محبت کا", "میں نہیں"]
    
    print("Step 1: Character-level tokenization")
    char_vocab = {}
    for word in corpus:
        chars = list(word) + ['</w>']  # End-of-word marker
        char_word = ' '.join(chars)
        print(f"'{word}' → '{char_word}'")
        char_vocab[char_word] = char_vocab.get(char_word, 0) + 1
    
    print(f"\nInitial vocabulary: {char_vocab}")
    
    print("\nStep 2: Find most frequent pairs")
    pairs = {}
    for word, freq in char_vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    
    print(f"Character pairs: {pairs}")
    
    # Most frequent pair would be merged
    if pairs:
        most_frequent = max(pairs, key=pairs.get)
        print(f"Most frequent pair to merge: {most_frequent}")

# =============================================================================
# STEP 5: Understanding Training Loop
# =============================================================================
def training_loop_explanation():
    """
    Understand the training process
    """
    print("Training Loop Steps:")
    print("1. Forward Pass:")
    print("   - Encoder processes Urdu sentence")
    print("   - Decoder generates Roman Urdu word by word")
    print("   - Attention helps decoder focus on relevant parts")
    
    print("\n2. Loss Calculation:")
    print("   - Compare predicted words with actual Roman Urdu")
    print("   - Use Cross-Entropy Loss")
    print("   - Ignore padding tokens")
    
    print("\n3. Backward Pass:")
    print("   - Calculate gradients")
    print("   - Update model parameters")
    print("   - Clip gradients to prevent explosion")
    
    print("\n4. Teacher Forcing:")
    print("   - During training: use real target words as input")
    print("   - During inference: use model's own predictions")
    print("   - Gradually reduce teacher forcing ratio")

def main():
    """
    Run all learning examples
    """
    print("=== NEURAL MACHINE TRANSLATION LEARNING GUIDE ===\n")
    
    print("1. EMBEDDING EXAMPLE:")
    create_embedding_example()
    print("\n" + "="*50 + "\n")
    
    print("2. LSTM EXAMPLE:")
    create_lstm_example()
    print("\n" + "="*50 + "\n")
    
    print("3. ATTENTION EXAMPLE:")
    create_attention_example()
    print("\n" + "="*50 + "\n")
    
    print("4. BPE TOKENIZATION EXAMPLE:")
    bpe_learning_example()
    print("\n" + "="*50 + "\n")
    
    print("5. TRAINING PROCESS:")
    training_loop_explanation()
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
