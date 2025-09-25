# =============================================================================
# GOOGLE COLAB DATASET FIX
# Run this cell first to extract and set up the dataset correctly
# =============================================================================

import os
import zipfile
import shutil
from tqdm import tqdm
import re
import unicodedata
import random
from typing import List, Tuple

def extract_and_setup_dataset():
    """
    Extract the dataset zip file and find the correct path
    """
    print("🔧 FIXING DATASET FOR GOOGLE COLAB")
    print("=" * 50)
    
    # Known zip path
    zip_path = '/content/urdu_ghazals_rekhta/dataset/dataset.zip'
    extract_to = '/content/dataset_extracted'
    
    print(f"📦 Extracting {zip_path}")
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found at {zip_path}")
        # Try alternative locations
        alt_paths = [
            '/content/urdu_ghazals_rekhta/dataset.zip',
            '/content/dataset/dataset.zip', 
            '/content/dataset.zip'
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                zip_path = alt_path
                print(f"✅ Found zip at: {zip_path}")
                break
        else:
            print("❌ No zip file found! Please check the path.")
            return None
    
    # Extract the zip file
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✅ Extracted to: {extract_to}")
    
    # Find the actual dataset directory with poets
    dataset_dir = None
    for root, dirs, files in os.walk(extract_to):
        # Look for poet directories
        poet_dirs = [d for d in dirs if any(poet in d for poet in ['mirza-ghalib', 'ahmad-faraz', 'allama-iqbal'])]
        if poet_dirs:
            dataset_dir = root
            print(f"🎯 Found dataset directory: {dataset_dir}")
            print(f"📚 Sample poets: {poet_dirs[:3]}...")
            break
    
    if not dataset_dir:
        print("❌ Could not find poet directories in extracted files")
        return None
    
    return dataset_dir

# Text cleaning classes
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

def load_dataset(data_path):
    """
    Load Urdu-Roman Urdu parallel corpus
    """
    print(f"📖 Loading dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset path does not exist: {data_path}")
        return [], []
    
    urdu_texts = []
    roman_texts = []
    
    # Get all poet directories
    poets = [d for d in os.listdir(data_path) 
             if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('.')]
    
    print(f"📚 Found {len(poets)} poets")
    
    for poet in tqdm(poets, desc="Loading poets"):
        poet_path = os.path.join(data_path, poet)
        urdu_path = os.path.join(poet_path, 'ur')
        english_path = os.path.join(poet_path, 'en')
        
        if os.path.exists(urdu_path) and os.path.exists(english_path):
            urdu_files = set(os.listdir(urdu_path))
            english_files = set(os.listdir(english_path))
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
                    
                    # Pair lines
                    min_lines = min(len(urdu_lines), len(roman_lines))
                    for i in range(min_lines):
                        if urdu_lines[i] and roman_lines[i]:
                            urdu_texts.append(urdu_lines[i])
                            roman_texts.append(roman_lines[i])
                            
                except Exception as e:
                    continue
    
    print(f"✅ Loaded {len(urdu_texts)} sentence pairs")
    if len(urdu_texts) > 0:
        print(f"📝 Sample - Urdu: {urdu_texts[0]}")
        print(f"📝 Sample - Roman: {roman_texts[0]}")
    
    return urdu_texts, roman_texts

def clean_and_split_data(urdu_texts: List[str], roman_texts: List[str], 
                        train_ratio: float = 0.5, val_ratio: float = 0.25, test_ratio: float = 0.25):
    """
    Clean data and split into train/val/test sets
    """
    if len(urdu_texts) == 0:
        print("❌ No data to process!")
        return [], [], []
    
    cleaner = TextCleaner()
    
    print("🧹 Cleaning texts...")
    cleaned_urdu = [cleaner.clean_urdu(text) for text in tqdm(urdu_texts, desc="Cleaning Urdu")]
    cleaned_roman = [cleaner.clean_roman(text) for text in tqdm(roman_texts, desc="Cleaning Roman")]
    
    # Filter valid pairs
    valid_pairs = []
    for u, r in zip(cleaned_urdu, cleaned_roman):
        if u and r and 2 <= len(u.split()) <= 50 and 2 <= len(r.split()) <= 50:
            valid_pairs.append((u, r))
    
    print(f"✅ After filtering: {len(valid_pairs)} valid pairs")
    
    if len(valid_pairs) == 0:
        print("❌ No valid pairs after filtering!")
        return [], [], []
    
    # Shuffle and split
    random.shuffle(valid_pairs)
    total = len(valid_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]
    
    print(f"📊 Data splits:")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Validation: {len(val_pairs)} pairs") 
    print(f"  Test: {len(test_pairs)} pairs")
    
    return train_pairs, val_pairs, test_pairs

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("🚀 Setting up dataset for Google Colab...")

# Extract and find dataset
dataset_path = extract_and_setup_dataset()

if dataset_path:
    print(f"\n✅ SUCCESS! Dataset is ready!")
    print(f"📍 Dataset path: {dataset_path}")
    
    # Test loading a small sample
    print(f"\n🧪 Testing dataset loading...")
    urdu_texts, roman_texts = load_dataset(dataset_path)
    
    if len(urdu_texts) > 0:
        print(f"🎉 Dataset loading successful!")
        print(f"📊 Total pairs loaded: {len(urdu_texts)}")
        
        # Set the global dataset_path variable for later use
        globals()['dataset_path'] = dataset_path
        
        print(f"\n📝 READY TO PROCEED!")
        print(f"📍 Use this path in your training: dataset_path = '{dataset_path}'")
        
    else:
        print(f"❌ No data loaded. Please check the dataset structure.")
else:
    print(f"\n❌ Failed to set up dataset. Please check the zip file location.")
