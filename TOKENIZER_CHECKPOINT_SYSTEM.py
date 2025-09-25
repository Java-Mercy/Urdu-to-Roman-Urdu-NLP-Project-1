"""
TOKENIZER CHECKPOINT SYSTEM - Save 15+ minutes of BPE training time!
This system saves and loads trained tokenizers to avoid retraining every time.
"""

import pickle
import os
import hashlib
from typing import List, Tuple, Dict, Any

class TokenizerCheckpoint:
    """System to save and load tokenizer checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "/content/tokenizer_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _get_data_hash(self, train_pairs: List[Tuple], vocab_sizes: Dict[str, int]) -> str:
        """Generate hash for data to detect changes"""
        data_str = f"{len(train_pairs)}_{vocab_sizes['src']}_{vocab_sizes['tgt']}"
        # Add hash of first few pairs to detect content changes
        sample_pairs = str(train_pairs[:10]) if len(train_pairs) > 10 else str(train_pairs)
        data_str += hashlib.md5(sample_pairs.encode()).hexdigest()[:8]
        return data_str
    
    def save_tokenizers(self, src_tokenizer, tgt_tokenizer, train_pairs: List[Tuple], 
                       vocab_sizes: Dict[str, int], config: Dict[str, Any]):
        """Save tokenizers with metadata"""
        data_hash = self._get_data_hash(train_pairs, vocab_sizes)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"tokenizers_{data_hash}.pkl")
        
        checkpoint_data = {
            'src_tokenizer': src_tokenizer,
            'tgt_tokenizer': tgt_tokenizer,
            'vocab_sizes': vocab_sizes,
            'config': config,
            'data_hash': data_hash,
            'num_pairs': len(train_pairs)
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"💾 Tokenizers saved to: {checkpoint_path}")
        return checkpoint_path
    
    def load_tokenizers(self, train_pairs: List[Tuple], vocab_sizes: Dict[str, int]) -> Tuple[Any, Any, bool]:
        """Load tokenizers if checkpoint exists"""
        data_hash = self._get_data_hash(train_pairs, vocab_sizes)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"tokenizers_{data_hash}.pkl")
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                print(f"📂 Loading tokenizers from: {checkpoint_path}")
                print(f"📊 Checkpoint info: {checkpoint_data['num_pairs']} pairs, "
                      f"src_vocab={checkpoint_data['vocab_sizes']['src']}, "
                      f"tgt_vocab={checkpoint_data['vocab_sizes']['tgt']}")
                
                return (checkpoint_data['src_tokenizer'], 
                       checkpoint_data['tgt_tokenizer'], 
                       True)
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
                return None, None, False
        else:
            print(f"🔍 No checkpoint found for data hash: {data_hash}")
            return None, None, False
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith("tokenizers_") and file.endswith(".pkl"):
                checkpoints.append(file)
        
        if checkpoints:
            print("📋 Available tokenizer checkpoints:")
            for cp in checkpoints:
                print(f"  - {cp}")
        else:
            print("📋 No tokenizer checkpoints found")
        
        return checkpoints

# Enhanced version of the main training script with checkpoint system
def create_tokenizers_with_checkpoint(train_pairs, config, checkpoint_system):
    """Create tokenizers with checkpoint support"""
    print("\n🔧 Creating tokenizers with checkpoint system...")
    
    vocab_sizes = {
        'src': config['src_vocab_size'],
        'tgt': config['tgt_vocab_size']
    }
    
    # Try to load from checkpoint first
    src_tokenizer, tgt_tokenizer, loaded = checkpoint_system.load_tokenizers(train_pairs, vocab_sizes)
    
    if loaded:
        print("✅ Tokenizers loaded from checkpoint - saving 15+ minutes!")
        return src_tokenizer, tgt_tokenizer
    else:
        print("🔄 No checkpoint found, training new tokenizers...")
        
        # Import the tokenizer creation function from your main script
        from FINAL_ERROR_FREE_COLAB import create_tokenizers
        
        # Train new tokenizers
        src_tokenizer, tgt_tokenizer = create_tokenizers(
            train_pairs,
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size']
        )
        
        # Save checkpoint for next time
        checkpoint_system.save_tokenizers(src_tokenizer, tgt_tokenizer, train_pairs, vocab_sizes, config)
        
        return src_tokenizer, tgt_tokenizer

# Quick dataset subset for testing
def create_minimal_dataset(dataset_path, max_poets=3, max_files_per_poet=5):
    """Create a minimal dataset for quick testing"""
    print(f"🚀 Creating minimal dataset (max {max_poets} poets, {max_files_per_poet} files each)...")
    
    from FINAL_ERROR_FREE_COLAB import load_dataset
    
    # Load full dataset
    urdu_texts, roman_texts = load_dataset(dataset_path)
    
    # Take only a subset for quick testing
    subset_size = min(1000, len(urdu_texts))  # Max 1000 pairs for quick testing
    urdu_subset = urdu_texts[:subset_size]
    roman_subset = roman_texts[:subset_size]
    
    print(f"📊 Minimal dataset: {len(urdu_subset)} pairs (from {len(urdu_texts)} total)")
    
    return urdu_subset, roman_subset

# Enhanced training script with all optimizations
def run_optimized_training():
    """Run training with all time-saving optimizations"""
    print("🚀 OPTIMIZED TRAINING - Maximum Time Savings!")
    print("=" * 60)
    
    # Configuration
    config = {
        'seed': 42,
        'embed_dim': 256,
        'hidden_dim': 512,
        'encoder_layers': 2,
        'decoder_layers': 4,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'src_vocab_size': 8000,
        'tgt_vocab_size': 8000,
        'num_epochs': 10
    }
    
    # Initialize checkpoint system
    checkpoint_system = TokenizerCheckpoint()
    
    # Check for existing checkpoints
    checkpoint_system.list_checkpoints()
    
    # Find dataset
    dataset_path = None
    if os.path.exists('/content/dataset_extracted'):
        for root, dirs, files in os.walk('/content/dataset_extracted'):
            if any(poet in dirs for poet in ['mirza-ghalib', 'ahmad-faraz', 'allama-iqbal']):
                dataset_path = root
                break
    
    if not dataset_path:
        print("❌ Dataset not found!")
        return
    
    print(f"✅ Using dataset: {dataset_path}")
    
    # Create minimal dataset for quick testing
    urdu_texts, roman_texts = create_minimal_dataset(dataset_path)
    
    # Clean and split data
    from FINAL_ERROR_FREE_COLAB import clean_and_split_data
    train_pairs, val_pairs, test_pairs = clean_and_split_data(
        urdu_texts, roman_texts,
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )
    
    if len(train_pairs) == 0:
        print("❌ No training data available!")
        return
    
    print(f"✅ Ready with {len(train_pairs)} training pairs")
    
    # Create tokenizers with checkpoint system
    src_tokenizer, tgt_tokenizer = create_tokenizers_with_checkpoint(
        train_pairs, config, checkpoint_system
    )
    
    # Rest of training continues...
    print("🎯 Tokenizers ready - proceeding with model creation and training...")

if __name__ == "__main__":
    run_optimized_training()
