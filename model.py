import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import random

class BiLSTMEncoder(nn.Module):
    """
    2-layer Bidirectional LSTM Encoder
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))
        
        if lengths is not None:
            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)
            output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # output: (batch_size, seq_len, hidden_dim * 2)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        # cell: (num_layers * 2, batch_size, hidden_dim)
        
        # Combine forward and backward hidden states
        # Take the last layer's hidden states
        hidden_fwd = hidden[-2]  # Forward direction
        hidden_bwd = hidden[-1]  # Backward direction
        final_hidden = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        cell_fwd = cell[-2]
        cell_bwd = cell[-1]
        final_cell = torch.cat([cell_fwd, cell_bwd], dim=1)
        
        return output, (final_hidden, final_cell)

class LSTMDecoder(nn.Module):
    """
    4-layer LSTM Decoder with Attention
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 encoder_hidden_dim: int, num_layers: int = 4, dropout: float = 0.3):
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Project encoder hidden state to decoder hidden state size
        self.hidden_projection = nn.Linear(encoder_hidden_dim, hidden_dim)
        self.cell_projection = nn.Linear(encoder_hidden_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = AttentionMechanism(hidden_dim, encoder_hidden_dim * 2)
        
        # LSTM layers
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim * 2, hidden_dim, 
                           num_layers, batch_first=True, dropout=dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        # input_token: (batch_size, 1)
        # hidden: (num_layers, batch_size, hidden_dim)
        # cell: (num_layers, batch_size, hidden_dim)
        # encoder_outputs: (batch_size, seq_len, encoder_hidden_dim * 2)
        
        embedded = self.dropout(self.embedding(input_token))
        # embedded: (batch_size, 1, embed_dim)
        
        # Apply attention
        context, attention_weights = self.attention(
            hidden[-1].unsqueeze(1), encoder_outputs, mask)
        # context: (batch_size, 1, encoder_hidden_dim * 2)
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context], dim=2)
        # lstm_input: (batch_size, 1, embed_dim + encoder_hidden_dim * 2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch_size, 1, hidden_dim)
        
        # Project to vocabulary size
        output = self.output_projection(output)
        # output: (batch_size, 1, vocab_size)
        
        return output, hidden, cell, attention_weights

class AttentionMechanism(nn.Module):
    """
    Additive (Bahdanau) Attention Mechanism
    """
    
    def __init__(self, decoder_hidden_dim: int, encoder_hidden_dim: int):
        super(AttentionMechanism, self).__init__()
        self.decoder_projection = nn.Linear(decoder_hidden_dim, encoder_hidden_dim)
        self.encoder_projection = nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        self.attention_projection = nn.Linear(encoder_hidden_dim, 1)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: (batch_size, 1, decoder_hidden_dim)
        # encoder_outputs: (batch_size, seq_len, encoder_hidden_dim)
        
        batch_size, seq_len, encoder_dim = encoder_outputs.size()
        
        # Project decoder hidden state
        decoder_proj = self.decoder_projection(decoder_hidden)
        # decoder_proj: (batch_size, 1, encoder_hidden_dim)
        
        # Project encoder outputs
        encoder_proj = self.encoder_projection(encoder_outputs)
        # encoder_proj: (batch_size, seq_len, encoder_hidden_dim)
        
        # Calculate attention scores
        energy = torch.tanh(decoder_proj + encoder_proj)
        # energy: (batch_size, seq_len, encoder_hidden_dim)
        
        attention_scores = self.attention_projection(energy).squeeze(2)
        # attention_scores: (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights: (batch_size, seq_len)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # context: (batch_size, 1, encoder_hidden_dim)
        
        return context, attention_weights

class Seq2SeqModel(nn.Module):
    """
    Complete Sequence-to-Sequence Model with BiLSTM Encoder and LSTM Decoder
    """
    
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
            encoder_hidden_dim=hidden_dim * 2,  # Bidirectional
            num_layers=decoder_layers,
            dropout=dropout
        )
        
        self.tgt_vocab_size = tgt_vocab_size
        
    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # Encoder
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Create mask for encoder outputs
        if src_lengths is not None:
            mask = torch.zeros(batch_size, src.size(1), device=src.device)
            for i, length in enumerate(src_lengths):
                mask[i, :length] = 1
        else:
            mask = torch.ones(batch_size, src.size(1), device=src.device)
        
        # Initialize decoder hidden states
        decoder_hidden = self.decoder.hidden_projection(hidden).unsqueeze(0)
        decoder_cell = self.decoder.cell_projection(cell).unsqueeze(0)
        
        # Repeat for all decoder layers
        decoder_hidden = decoder_hidden.repeat(self.decoder.num_layers, 1, 1)
        decoder_cell = decoder_cell.repeat(self.decoder.num_layers, 1, 1)
        
        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)
        
        # First input to decoder is SOS token
        input_token = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        for t in range(1, tgt_len):
            output, decoder_hidden, decoder_cell, _ = self.decoder(
                input_token, decoder_hidden, decoder_cell, encoder_outputs, mask)
            
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing
            if random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t].unsqueeze(1)
            else:
                input_token = output.argmax(2)
        
        return outputs
    
    def inference(self, src, src_tokenizer, tgt_tokenizer, max_length=50):
        """
        Inference method for generating translations
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # Encoder
            encoder_outputs, (hidden, cell) = self.encoder(src)
            
            # Create mask
            mask = torch.ones(batch_size, src.size(1), device=device)
            
            # Initialize decoder
            decoder_hidden = self.decoder.hidden_projection(hidden).unsqueeze(0)
            decoder_cell = self.decoder.cell_projection(cell).unsqueeze(0)
            decoder_hidden = decoder_hidden.repeat(self.decoder.num_layers, 1, 1)
            decoder_cell = decoder_cell.repeat(self.decoder.num_layers, 1, 1)
            
            # Start with SOS token
            sos_id = tgt_tokenizer.vocab['<sos>']
            eos_id = tgt_tokenizer.vocab['<eos>']
            
            input_token = torch.full((batch_size, 1), sos_id, device=device)
            
            outputs = []
            
            for _ in range(max_length):
                output, decoder_hidden, decoder_cell, _ = self.decoder(
                    input_token, decoder_hidden, decoder_cell, encoder_outputs, mask)
                
                predicted = output.argmax(2)
                outputs.append(predicted)
                
                input_token = predicted
                
                # Stop if all sequences have generated EOS
                if (predicted == eos_id).all():
                    break
            
            # Concatenate outputs
            if outputs:
                outputs = torch.cat(outputs, dim=1)
            else:
                outputs = torch.empty(batch_size, 0, device=device)
            
            return outputs
