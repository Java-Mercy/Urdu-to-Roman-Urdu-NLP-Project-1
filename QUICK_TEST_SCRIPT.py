"""
QUICK TEST SCRIPT - Save 20 minutes of tokenization time!
This script creates a minimal test to validate model architecture without full dataset processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from collections import Counter
import re

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Quick model validation with synthetic data
def create_synthetic_data():
    """Create small synthetic dataset for quick testing"""
    print("🚀 Creating synthetic data for quick testing...")
    
    # Create small vocabulary
    src_vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + [f'urdu_{i}' for i in range(100)]
    tgt_vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + [f'roman_{i}' for i in range(100)]
    
    # Create synthetic sentence pairs
    synthetic_pairs = []
    for i in range(50):  # Only 50 pairs for quick testing
        src_len = np.random.randint(5, 15)
        tgt_len = np.random.randint(5, 15)
        
        src_sentence = [np.random.randint(4, 104) for _ in range(src_len)]
        tgt_sentence = [np.random.randint(4, 104) for _ in range(tgt_len)]
        
        synthetic_pairs.append((src_sentence, tgt_sentence))
    
    print(f"✅ Created {len(synthetic_pairs)} synthetic pairs")
    return synthetic_pairs, src_vocab, tgt_vocab

class QuickTokenizer:
    """Quick tokenizer for testing - no BPE training needed"""
    def __init__(self, vocab):
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(vocab)
    
    def encode(self, sentence):
        return sentence  # Already tokenized
    
    def decode(self, tokens):
        return [self.idx_to_word.get(token, '<UNK>') for token in tokens]
    
    def get_vocab_size(self):
        return self.vocab_size

# Import the fixed model classes
class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        
        if lengths is not None:
            # FIX: Move lengths to CPU
            lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
            output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # Handle bidirectional LSTM outputs
        hidden_fwd = hidden[0::2]  # Forward direction
        hidden_bwd = hidden[1::2]  # Backward direction
        cell_fwd = cell[0::2]
        cell_bwd = cell[1::2]
        
        # Concatenate forward and backward
        final_hidden = torch.cat([hidden_fwd, hidden_bwd], dim=2)
        final_cell = torch.cat([cell_fwd, cell_bwd], dim=2)
        
        return output, (final_hidden, final_cell)

class AttentionMechanism(nn.Module):
    def __init__(self, decoder_hidden_dim: int, encoder_hidden_dim: int):
        super(AttentionMechanism, self).__init__()
        self.decoder_projection = nn.Linear(decoder_hidden_dim, encoder_hidden_dim)
        self.encoder_projection = nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        self.attention_projection = nn.Linear(encoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, encoder_dim = encoder_outputs.size()

        # decoder_hidden shape: (num_layers, batch_size, hidden_dim)
        # We need: (batch_size, 1, hidden_dim) for attention
        decoder_hidden_proj = decoder_hidden[-1].unsqueeze(1)  # Take last layer, add seq dim

        # Project decoder hidden to encoder dimension
        decoder_proj = self.decoder_projection(decoder_hidden_proj)  # (batch_size, 1, encoder_dim)

        # Broadcast decoder projection to match encoder sequence length
        decoder_proj = decoder_proj.repeat(1, seq_len, 1)  # (batch_size, seq_len, encoder_dim)

        # Project encoder outputs to same dimension
        encoder_proj = self.encoder_projection(encoder_outputs)  # (batch_size, seq_len, encoder_dim)

        # Calculate attention scores
        energy = torch.tanh(decoder_proj + encoder_proj)  # (batch_size, seq_len, encoder_dim)
        attention_scores = self.attention_projection(energy).squeeze(2)  # (batch_size, seq_len)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)

        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, encoder_dim)

        return context, attention_weights

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 encoder_hidden_dim: int, num_layers: int = 4, dropout: float = 0.3):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.hidden_projection = nn.Linear(encoder_hidden_dim, hidden_dim)
        self.cell_projection = nn.Linear(encoder_hidden_dim, hidden_dim)
        
        # Fix: Use the actual encoder hidden dimension (encoder_hidden_dim is already hidden_dim * 2)
        self.attention = AttentionMechanism(hidden_dim, encoder_hidden_dim)
        
        # Fix: LSTM input should be embed_dim + encoder_hidden_dim (not *2)
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim, hidden_dim, 
                           num_layers, batch_first=True, dropout=dropout)
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(input_token))
        
        # Fix: Pass the full hidden state to attention, not just the last layer
        context, attention_weights = self.attention(hidden, encoder_outputs, mask)
        
        # Fix: Ensure context has the right shape for concatenation
        lstm_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.output_projection(output)
        
        return output, hidden, cell, attention_weights

class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 embed_dim: int = 256, hidden_dim: int = 512, 
                 encoder_layers: int = 2, decoder_layers: int = 4,
                 dropout: float = 0.3):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = BiLSTMEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout
        )
        
        self.decoder = LSTMDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_hidden_dim=hidden_dim * 2,  # This is correct - encoder outputs hidden_dim * 2
            num_layers=decoder_layers,
            dropout=dropout
        )
        
        self.tgt_vocab_size = tgt_vocab_size
        
    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Project encoder hidden states to decoder dimensions
        hidden = self.decoder.hidden_projection(hidden)
        cell = self.decoder.cell_projection(cell)
        
        # Create mask for encoder outputs
        if src_lengths is not None:
            mask = torch.arange(src.size(1)).expand(batch_size, src.size(1)).to(src.device) < src_lengths.unsqueeze(1)
        else:
            mask = torch.ones(batch_size, src.size(1), dtype=torch.bool).to(src.device)
        
        outputs = []
        input_token = tgt[:, 0:1]  # Start with SOS token
        
        for t in range(1, tgt_len):
            output, decoder_hidden, decoder_cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask)
            
            outputs.append(output)
            
            # Teacher forcing
            if np.random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t:t+1]
            else:
                input_token = output.argmax(dim=-1)
            
            hidden, cell = decoder_hidden, decoder_cell
        
        return torch.cat(outputs, dim=1)

def quick_model_test():
    """Quick test to validate model architecture"""
    print("🧪 QUICK MODEL ARCHITECTURE TEST")
    print("=" * 50)
    
    # Create synthetic data
    synthetic_pairs, src_vocab, tgt_vocab = create_synthetic_data()
    
    # Create tokenizers
    src_tokenizer = QuickTokenizer(src_vocab)
    tgt_tokenizer = QuickTokenizer(tgt_vocab)
    
    print(f"📊 Source vocab size: {src_tokenizer.get_vocab_size()}")
    print(f"📊 Target vocab size: {tgt_tokenizer.get_vocab_size()}")
    
    # Create model
    print("\n🏗️ Creating model...")
    model = Seq2SeqModel(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        embed_dim=128,  # Smaller for quick test
        hidden_dim=256,  # Smaller for quick test
        encoder_layers=2,
        decoder_layers=4,
        dropout=0.3
    )
    
    print(f"📈 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    batch_size = 4
    src_len = 10
    tgt_len = 8
    
    # Create dummy input
    src = torch.randint(4, 104, (batch_size, src_len))
    tgt = torch.randint(4, 104, (batch_size, tgt_len))
    src_lengths = torch.randint(5, src_len+1, (batch_size,))
    
    try:
        start_time = time.time()
        outputs = model(src, tgt, src_lengths)
        end_time = time.time()
        
        print(f"✅ Forward pass successful!")
        print(f"⏱️ Time taken: {end_time - start_time:.3f} seconds")
        print(f"📐 Output shape: {outputs.shape}")
        print(f"📐 Expected shape: ({batch_size}, {tgt_len-1}, {tgt_tokenizer.get_vocab_size()})")
        
        if outputs.shape == (batch_size, tgt_len-1, tgt_tokenizer.get_vocab_size()):
            print("🎉 MODEL ARCHITECTURE IS CORRECT!")
            return True
        else:
            print("❌ Output shape mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def main():
    print("🚀 QUICK TEST SCRIPT - Save 20 minutes!")
    print("This script validates your model architecture in seconds instead of minutes.")
    print("=" * 70)
    
    success = quick_model_test()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Your model architecture is working correctly.")
        print("💡 You can now run the full training script with confidence!")
        print("⏰ This test took seconds instead of 20 minutes!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ FAILED! There are still issues with the model architecture.")
        print("🔧 Please fix the errors before running the full training.")
        print("=" * 70)

if __name__ == "__main__":
    main()
