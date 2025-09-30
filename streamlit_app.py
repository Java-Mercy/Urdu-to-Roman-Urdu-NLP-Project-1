import streamlit as st
import torch
import pickle
import os
import re
import unicodedata
from typing import List, Tuple, Dict, Any
import warnings
from huggingface_hub import hf_hub_download, snapshot_download
warnings.filterwarnings('ignore')

# Note: torchtext is not needed for the deployed version
# All tokenization is handled by our custom BPE implementation

# Set page config
st.set_page_config(
    page_title="Urdu to Roman Urdu Translator",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model and tokenizer classes (copied from your notebook)
class TextCleaner:
    @staticmethod
    def clean_urdu(text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s۔،؍؎؏؟!]', '', text)
        return text.strip()
    
    @staticmethod
    def clean_roman(text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-ZāīūĀĪŪñṇṛṭḍṣġḥḳẓẕ\s\'\-\.]', '', text)
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

# Model classes (simplified versions)
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
        
        # Combine bidirectional hidden states
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

# Load model function
@st.cache_resource
def load_model_and_tokenizers():
    """Load model and tokenizers from Hugging Face with caching"""
    try:
        # Hugging Face repository - UPDATE THIS WITH YOUR ACTUAL REPO
        repo_id = "tahir-next/Urdu-RomanUrdu"  # Replace with your actual HF repo
            
        # Single model files
        model_filename = "best_model_exp_3.pth"  # Your trained model file
        tokenizer_filename = "exp_3_tokenizers.pkl"  # Your tokenizer file
        
        # Get Hugging Face token (for private repos)
        hf_token = None
        try:
            # Try to get token from Streamlit secrets
            hf_token = st.secrets["HF_TOKEN"]
        except:
            # If no token in secrets, try environment variable
            hf_token = os.getenv("HF_TOKEN")
        
        # Download model file from Hugging Face
        with st.spinner("🔄 Downloading model from Hugging Face..."):
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename,
                cache_dir="./hf_cache",
                token=hf_token
            )
        
        # Download tokenizer file from Hugging Face
        with st.spinner("🔄 Downloading tokenizers from Hugging Face..."):
            tokenizer_path = hf_hub_download(
                repo_id=repo_id,
                filename=tokenizer_filename,
                cache_dir="./hf_cache",
                token=hf_token
            )
        
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
        model = Seq2SeqModel(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dropout=0.3
        )
        
        # Load weights
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load tokenizers
        with open(tokenizer_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
            src_tokenizer = tokenizer_data['src_tokenizer']
            tgt_tokenizer = tokenizer_data['tgt_tokenizer']
        
        return model, src_tokenizer, tgt_tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Translation function
def translate_urdu_poetry(model, src_tokenizer, tgt_tokenizer, urdu_text, max_length=100):
    """Translate Urdu text to Roman Urdu"""
    if not model or not src_tokenizer or not tgt_tokenizer:
        return "Error: Model not loaded properly"
    
    model.eval()
    
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
        for step in range(max_length):
            outputs = model(src_tensor, target_tensor, src_lengths, teacher_forcing_ratio=0.0)
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
        final_outputs = model(src_tensor, target_tensor, src_lengths, teacher_forcing_ratio=0.0)
        probabilities = torch.softmax(final_outputs, dim=-1)
        confidence = torch.mean(torch.max(probabilities, dim=-1)[0]).item()
    
    return {
        'translation': generated_text,
        'confidence': confidence,
        'tokens_generated': len(target_sequence)
    }

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🌙 Urdu to Roman Urdu Translator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Neural Machine Translation for Urdu Poetry</p>', unsafe_allow_html=True)
    
    # Auto-load model from Hugging Face
    if 'model' not in st.session_state:
        with st.spinner("🔄 Loading model from Hugging Face..."):
            model, src_tokenizer, tgt_tokenizer = load_model_and_tokenizers()
            if model:
                st.session_state.model = model
                st.session_state.src_tokenizer = src_tokenizer
                st.session_state.tgt_tokenizer = tgt_tokenizer
                st.success("✅ Model loaded successfully from Hugging Face!")
            else:
                st.error("❌ Failed to load model from Hugging Face")
                st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📝 Input Urdu Text</h2>', unsafe_allow_html=True)
        
        # Text input
        urdu_text = st.text_area(
            "Enter Urdu poetry:",
            height=200,
            placeholder="یہ نہ تھی ہماری قسمت کہ وصال یار ہوتا"
        )
        
        # Example poems
        st.markdown("**Example Poems:**")
        example_poems = [
            "یہ نہ تھی ہماری قسمت کہ وصال یار ہوتا",
            "اگر اپنا کہا آپ ہی سمجھتے تو کیا کہتے",
            "دل سے جو بات نکلتی ہے اثر رکھتی ہے",
            "ہم کو معلوم ہے جنت کی حقیقت لیکن",
            "عشق میں غم کا مزہ بھی نہیں آتا"
        ]
        
        for i, poem in enumerate(example_poems):
            if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                st.session_state.example_text = poem
                st.rerun()
        
        if 'example_text' in st.session_state:
            urdu_text = st.session_state.example_text
            del st.session_state.example_text
    
    with col2:
        st.markdown('<h2 class="sub-header">📤 Translation Result</h2>', unsafe_allow_html=True)
        
        if urdu_text:
            if st.button("🚀 Translate", type="primary"):
                with st.spinner("Translating..."):
                    result = translate_urdu_poetry(
                        st.session_state.model,
                        st.session_state.src_tokenizer,
                        st.session_state.tgt_tokenizer,
                        urdu_text
                    )
                    
                    if isinstance(result, dict):
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"**Translation:** {result['translation']}")
                        st.markdown(f"**Confidence:** {result['confidence']:.3f}")
                        st.markdown(f"**Tokens Generated:** {result['tokens_generated']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(result)
        else:
            st.info("Enter Urdu text to see translation")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Built with ❤️ using PyTorch and Streamlit</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()