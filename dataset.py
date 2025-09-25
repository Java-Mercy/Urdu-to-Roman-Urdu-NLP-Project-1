import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import random

class TranslationDataset(Dataset):
    """
    Dataset class for Urdu to Roman Urdu translation
    """
    
    def __init__(self, pairs: List[Tuple[str, str]], src_tokenizer, tgt_tokenizer):
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
        # Tokenize all pairs
        self.tokenized_pairs = []
        for src_text, tgt_text in pairs:
            src_tokens = src_tokenizer.encode(src_text)
            tgt_tokens = tgt_tokenizer.encode(f"<sos> {tgt_text} <eos>")
            
            if len(src_tokens) > 0 and len(tgt_tokens) > 0:
                self.tokenized_pairs.append((src_tokens, tgt_tokens))
    
    def __len__(self):
        return len(self.tokenized_pairs)
    
    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.tokenized_pairs[idx]
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens, dtype=torch.long),
            'src_len': len(src_tokens),
            'tgt_len': len(tgt_tokens)
        }

def collate_fn(batch):
    """
    Collate function for DataLoader to handle variable length sequences
    """
    src_sequences = [item['src'] for item in batch]
    tgt_sequences = [item['tgt'] for item in batch]
    src_lengths = [item['src_len'] for item in batch]
    tgt_lengths = [item['tgt_len'] for item in batch]
    
    # Pad sequences
    src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_sequences, batch_first=True, padding_value=0)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': torch.tensor(src_lengths, dtype=torch.long),
        'tgt_lengths': torch.tensor(tgt_lengths, dtype=torch.long)
    }

def create_data_loaders(train_pairs, val_pairs, test_pairs, src_tokenizer, tgt_tokenizer, 
                       batch_size=32, num_workers=0):
    """
    Create DataLoaders for train, validation, and test sets
    """
    # Create datasets
    train_dataset = TranslationDataset(train_pairs, src_tokenizer, tgt_tokenizer)
    val_dataset = TranslationDataset(val_pairs, src_tokenizer, tgt_tokenizer)
    test_dataset = TranslationDataset(test_pairs, src_tokenizer, tgt_tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=num_workers
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Validation: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
