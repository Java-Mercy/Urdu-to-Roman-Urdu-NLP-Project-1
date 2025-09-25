import os
import re
import unicodedata
from tqdm import tqdm
from typing import List, Tuple
import random

class TextCleaner:
    """
    Text cleaning and normalization utilities
    """
    
    @staticmethod
    def clean_urdu(text: str) -> str:
        """
        Clean and normalize Urdu text
        """
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unwanted punctuation but keep essential ones
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s۔،؍؎؏؟!]', '', text)
        
        return text.strip()
    
    @staticmethod
    def clean_roman(text: str) -> str:
        """
        Clean and normalize Roman Urdu text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Keep only alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-ZāīūĀĪŪñṇṛṭḍṣġḥḳẓẕ\s\'\-\.]', '', text)
        
        return text.strip()
    
    @staticmethod
    def add_special_tokens(text: str, is_target: bool = False) -> str:
        """
        Add special tokens for sequence processing
        """
        if is_target:
            return f"<sos> {text} <eos>"
        return text

def load_dataset(data_path='dataset/dataset'):
    """
    Load Urdu-Roman Urdu parallel corpus from the dataset
    """
    urdu_texts = []
    roman_texts = []
    
    # Get all poet directories
    poets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Found {len(poets)} poets in dataset")
    
    for poet in tqdm(poets, desc="Loading poets"):
        poet_path = os.path.join(data_path, poet)
        urdu_path = os.path.join(poet_path, 'ur')
        english_path = os.path.join(poet_path, 'en')
        
        if os.path.exists(urdu_path) and os.path.exists(english_path):
            # Get all files in both directories
            urdu_files = set(os.listdir(urdu_path))
            english_files = set(os.listdir(english_path))
            
            # Find common files
            common_files = urdu_files.intersection(english_files)
            
            for file_name in common_files:
                try:
                    # Read Urdu text
                    with open(os.path.join(urdu_path, file_name), 'r', encoding='utf-8') as f:
                        urdu_content = f.read().strip()
                    
                    # Read Roman Urdu text
                    with open(os.path.join(english_path, file_name), 'r', encoding='utf-8') as f:
                        roman_content = f.read().strip()
                    
                    # Split into lines and pair them
                    urdu_lines = [line.strip() for line in urdu_content.split('\n') if line.strip()]
                    roman_lines = [line.strip() for line in roman_content.split('\n') if line.strip()]
                    
                    # Ensure same number of lines
                    min_lines = min(len(urdu_lines), len(roman_lines))
                    for i in range(min_lines):
                        if urdu_lines[i] and roman_lines[i]:
                            urdu_texts.append(urdu_lines[i])
                            roman_texts.append(roman_lines[i])
                            
                except Exception as e:
                    print(f"Error processing {poet}/{file_name}: {e}")
                    continue
    
    print(f"\nDataset loaded:")
    print(f"Total pairs: {len(urdu_texts)}")
    
    return urdu_texts, roman_texts

def clean_and_split_data(urdu_texts: List[str], roman_texts: List[str], 
                        train_ratio: float = 0.5, val_ratio: float = 0.25, test_ratio: float = 0.25):
    """
    Clean data and split into train/val/test sets
    """
    cleaner = TextCleaner()
    
    print("Cleaning texts...")
    cleaned_urdu = [cleaner.clean_urdu(text) for text in tqdm(urdu_texts, desc="Cleaning Urdu")]
    cleaned_roman = [cleaner.clean_roman(text) for text in tqdm(roman_texts, desc="Cleaning Roman")]
    
    # Filter out empty pairs and very short/long sequences
    valid_pairs = []
    for u, r in zip(cleaned_urdu, cleaned_roman):
        if u and r and 2 <= len(u.split()) <= 50 and 2 <= len(r.split()) <= 50:
            valid_pairs.append((u, r))
    
    print(f"After cleaning and filtering: {len(valid_pairs)} valid pairs")
    
    # Shuffle data
    random.shuffle(valid_pairs)
    
    # Split data
    total = len(valid_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Validation: {len(val_pairs)} pairs")
    print(f"  Test: {len(test_pairs)} pairs")
    
    return train_pairs, val_pairs, test_pairs
