import gradio as gr
import torch
import pickle
import os
import re
import unicodedata
import warnings
warnings.filterwarnings('ignore')

# Model and tokenizer classes
class TextCleaner:
    @staticmethod
    def clean_urdu(text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s۔،؍؎؏؟!]', '', text)
        return text.strip()

class BPETokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.vocab = {}
        self.merges = []
        self.special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        
    def encode(self, text):
        tokens = []
        words = text.split()
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab.get('<unk>', 1))
        return tokens
    
    def decode(self, token_ids):
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)
        
        text = ' '.join(tokens)
        text = text.replace('</w>', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_vocab_size(self):
        return len(self.vocab)

# Model classes (simplified)
class BiLSTMEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        
        if lengths is not None:
            lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
            output, (hidden, cell) = self.lstm(packed)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        final_hidden = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        cell_fwd = cell[-2]
        cell_bwd = cell[-1]
        final_cell = torch.cat([cell_fwd, cell_bwd], dim=1)
        
        return output, (final_hidden, final_cell)

class AttentionMechanism(torch.nn.Module):
    def __init__(self, decoder_hidden_dim: int, encoder_hidden_dim: int):
        super(AttentionMechanism, self).__init__()
        self.decoder_projection = torch.nn.Linear(decoder_hidden_dim, encoder_hidden_dim)
        self.encoder_projection = torch.nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        self.attention_projection = torch.nn.Linear(encoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, encoder_dim = encoder_outputs.size()
        decoder_hidden_proj = decoder_hidden[-1].unsqueeze(1)
        decoder_proj = self.decoder_projection(decoder_hidden_proj)
        decoder_proj = decoder_proj.repeat(1, seq_len, 1)
        encoder_proj = self.encoder_projection(encoder_outputs)
        energy = torch.tanh(decoder_proj + encoder_proj)
        attention_scores = self.attention_projection(energy).squeeze(2)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)

        return context, attention_weights

class LSTMDecoder(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 encoder_hidden_dim: int, num_layers: int = 4, dropout: float = 0.3):
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_projection = torch.nn.Linear(encoder_hidden_dim, hidden_dim)
        self.cell_projection = torch.nn.Linear(encoder_hidden_dim, hidden_dim)
        self.attention = AttentionMechanism(hidden_dim, encoder_hidden_dim)
        self.lstm = torch.nn.LSTM(embed_dim + encoder_hidden_dim, hidden_dim, 
                           num_layers, batch_first=True, dropout=dropout)
        self.output_projection = torch.nn.Linear(hidden_dim, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(input_token))
        context, attention_weights = self.attention(hidden, encoder_outputs, mask)
        lstm_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.output_projection(output)
        return output, hidden, cell, attention_weights

class Seq2SeqModel(torch.nn.Module):
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
            encoder_hidden_dim=hidden_dim * 2,
            num_layers=decoder_layers,
            dropout=dropout
        )
        
        self.tgt_vocab_size = tgt_vocab_size
        
    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        if src_lengths is not None:
            mask = torch.zeros(batch_size, src.size(1), device=src.device)
            for i, length in enumerate(src_lengths):
                mask[i, :length] = 1
        else:
            mask = torch.ones(batch_size, src.size(1), device=src.device)
        
        decoder_hidden = self.decoder.hidden_projection(hidden).unsqueeze(0)
        decoder_cell = self.decoder.cell_projection(cell).unsqueeze(0)
        
        decoder_hidden = decoder_hidden.repeat(self.decoder.num_layers, 1, 1)
        decoder_cell = decoder_cell.repeat(self.decoder.num_layers, 1, 1)
        
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)
        
        input_token = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            output, decoder_hidden, decoder_cell, _ = self.decoder(
                input_token, decoder_hidden, decoder_cell, encoder_outputs, mask)
            
            outputs[:, t] = output.squeeze(1)
            
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t].unsqueeze(1)
            else:
                input_token = output.argmax(2)
        
        return outputs

# Global variables for loaded model
loaded_model = None
src_tokenizer = None
tgt_tokenizer = None

def load_model_and_tokenizers(model_choice):
    """Load model and tokenizers based on choice"""
    global loaded_model, src_tokenizer, tgt_tokenizer
    
    try:
        # Model paths
        model_paths = {
            "Experiment 1 (Small)": "models/best_model_exp_1.pth",
            "Experiment 2 (Medium)": "models/best_model_exp_2.pth", 
            "Experiment 3 (Large)": "models/best_model_exp_3.pth"
        }
        
        tokenizer_paths = {
            "Experiment 1 (Small)": "tokenizers/exp_1_tokenizers.pkl",
            "Experiment 2 (Medium)": "tokenizers/exp_2_tokenizers.pkl",
            "Experiment 3 (Large)": "tokenizers/exp_3_tokenizers.pkl"
        }
        
        model_path = model_paths[model_choice]
        tokenizer_path = tokenizer_paths[model_choice]
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Extract model parameters
        src_vocab_size = state_dict['encoder.embedding.weight'].shape[0]
        tgt_vocab_size = state_dict['decoder.embedding.weight'].shape[0]
        embed_dim = state_dict['encoder.embedding.weight'].shape[1]
        hidden_dim = state_dict['encoder.lstm.weight_ih_l0'].shape[0] // 4
        
        # Count layers
        encoder_layers = max([int(k.split('_l')[1].split('_')[0]) for k in state_dict.keys() 
                             if 'encoder.lstm.weight_ih_l' in k and '_reverse' not in k]) + 1
        decoder_layers = max([int(k.split('_l')[1].split('_')[0]) for k in state_dict.keys() 
                             if 'decoder.lstm.weight_ih_l' in k]) + 1
        
        # Create model
        loaded_model = Seq2SeqModel(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dropout=0.3
        )
        
        # Load weights
        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()
        
        # Load tokenizers
        with open(tokenizer_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
            src_tokenizer = tokenizer_data['src_tokenizer']
            tgt_tokenizer = tokenizer_data['tgt_tokenizer']
        
        return f"✅ {model_choice} loaded successfully!"
        
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def translate_urdu_poetry(urdu_text, model_choice):
    """Translate Urdu text to Roman Urdu"""
    global loaded_model, src_tokenizer, tgt_tokenizer
    
    if not urdu_text.strip():
        return "Please enter Urdu text to translate.", 0.0
    
    # Load model if not loaded or different choice
    load_status = load_model_and_tokenizers(model_choice)
    if "Error" in load_status:
        return load_status, 0.0
    
    if not loaded_model or not src_tokenizer or not tgt_tokenizer:
        return "Model not loaded properly. Please try again.", 0.0
    
    try:
        # Clean input
        cleaner = TextCleaner()
        cleaned_urdu = cleaner.clean_urdu(urdu_text)
        
        # Tokenize
        src_tokens = src_tokenizer.encode(cleaned_urdu)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long)
        src_lengths = torch.tensor([len(src_tokens)], dtype=torch.long)
        
        # Initialize target
        sos_token = tgt_tokenizer.vocab.get('<sos>', 1)
        eos_token = tgt_tokenizer.vocab.get('<eos>', 2)
        
        target_sequence = [sos_token]
        target_tensor = torch.tensor([target_sequence], dtype=torch.long)
        
        # Generate translation
        with torch.no_grad():
            for step in range(100):  # max_length
                outputs = loaded_model(src_tensor, target_tensor, src_lengths, teacher_forcing_ratio=0.0)
                next_token_logits = outputs[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                
                if next_token == eos_token:
                    break
                
                target_sequence.append(next_token)
                target_tensor = torch.tensor([target_sequence], dtype=torch.long)
        
        # Decode
        generated_text = tgt_tokenizer.decode(target_sequence)
        
        # Calculate confidence
        with torch.no_grad():
            final_outputs = loaded_model(src_tensor, target_tensor, src_lengths, teacher_forcing_ratio=0.0)
            probabilities = torch.softmax(final_outputs, dim=-1)
            confidence = torch.mean(torch.max(probabilities, dim=-1)[0]).item()
        
        return generated_text, confidence
        
    except Exception as e:
        return f"Translation error: {str(e)}", 0.0

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="🌙 Urdu to Roman Urdu Translator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            font-size: 2.5rem;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .sub-header {
            text-align: center;
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="main-header">🌙 Urdu to Roman Urdu Translator</div>
        <div class="sub-header">Neural Machine Translation for Urdu Poetry</div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    choices=["Experiment 1 (Small)", "Experiment 2 (Medium)", "Experiment 3 (Large)"],
                    value="Experiment 2 (Medium)",
                    label="Select Model",
                    info="Choose the trained model for translation"
                )
                
                load_btn = gr.Button("🔄 Load Model", variant="secondary")
                load_status = gr.Textbox(label="Model Status", interactive=False)
                
                gr.Examples(
                    examples=[
                        "یہ نہ تھی ہماری قسمت کہ وصال یار ہوتا",
                        "اگر اپنا کہا آپ ہی سمجھتے تو کیا کہتے",
                        "دل سے جو بات نکلتی ہے اثر رکھتی ہے",
                        "ہم کو معلوم ہے جنت کی حقیقت لیکن",
                        "عشق میں غم کا مزہ بھی نہیں آتا"
                    ],
                    inputs=gr.Textbox(label="Urdu Text", placeholder="Enter Urdu poetry here..."),
                    label="Example Poems"
                )
            
            with gr.Column(scale=1):
                urdu_input = gr.Textbox(
                    label="Urdu Text",
                    placeholder="یہ نہ تھی ہماری قسمت کہ وصال یار ہوتا",
                    lines=5
                )
                
                translate_btn = gr.Button("🚀 Translate", variant="primary", size="lg")
                
                roman_output = gr.Textbox(
                    label="Roman Urdu Translation",
                    lines=3,
                    interactive=False
                )
                
                confidence_score = gr.Number(
                    label="Confidence Score",
                    precision=3,
                    interactive=False
                )
        
        # Event handlers
        load_btn.click(
            fn=load_model_and_tokenizers,
            inputs=[model_choice],
            outputs=[load_status]
        )
        
        translate_btn.click(
            fn=translate_urdu_poetry,
            inputs=[urdu_input, model_choice],
            outputs=[roman_output, confidence_score]
        )
        
        # Auto-load model on startup
        interface.load(
            fn=lambda: load_model_and_tokenizers("Experiment 2 (Medium)"),
            outputs=[load_status]
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; color: #666;">
            <p>Built with ❤️ using PyTorch and Gradio</p>
            <p>Neural Machine Translation for Urdu Poetry</p>
        </div>
        """)
    
    return interface

# Launch the app
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public link
        show_error=True
    )
