#!/usr/bin/env python3
"""
Fix dataset loading for Google Colab environment
This script extracts the zip file and sets up the correct dataset path
"""

import os
import zipfile
import shutil

def extract_dataset_zip():
    """
    Extract the dataset zip file in Google Colab
    """
    zip_path = '/content/urdu_ghazals_rekhta/dataset/dataset.zip'
    extract_to = '/content/dataset_extracted'
    
    print(f"🔍 Checking for dataset zip at: {zip_path}")
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found at {zip_path}")
        
        # Alternative paths to check
        alternative_paths = [
            '/content/urdu_ghazals_rekhta/dataset.zip',
            '/content/dataset/dataset.zip',
            '/content/dataset.zip'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"✅ Found alternative zip at: {alt_path}")
                zip_path = alt_path
                break
        else:
            print("❌ No dataset zip file found!")
            return None
    
    print(f"📦 Extracting {zip_path} to {extract_to}")
    
    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all files
            zip_ref.extractall(extract_to)
            print("✅ Extraction completed!")
            
            # List extracted contents
            extracted_items = os.listdir(extract_to)
            print(f"📁 Extracted items: {extracted_items}")
            
            # Find the actual dataset directory
            dataset_dir = None
            for item in extracted_items:
                item_path = os.path.join(extract_to, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains poet folders
                    subdirs = os.listdir(item_path)
                    if any(poet in subdirs for poet in ['mirza-ghalib', 'ahmad-faraz', 'allama-iqbal']):
                        dataset_dir = item_path
                        break
            
            if not dataset_dir:
                # If no poet folders found at first level, check deeper
                for root, dirs, files in os.walk(extract_to):
                    if any(poet in dirs for poet in ['mirza-ghalib', 'ahmad-faraz', 'allama-iqbal']):
                        dataset_dir = root
                        break
            
            if dataset_dir:
                print(f"🎯 Found dataset directory: {dataset_dir}")
                
                # Verify structure
                poets = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
                print(f"📚 Found {len(poets)} poets: {poets[:5]}...")
                
                # Check sample poet structure
                if poets:
                    sample_poet = poets[0]
                    poet_path = os.path.join(dataset_dir, sample_poet)
                    subdirs = os.listdir(poet_path)
                    print(f"📖 Sample poet '{sample_poet}' structure: {subdirs}")
                    
                    if 'ur' in subdirs and 'en' in subdirs:
                        ur_files = len(os.listdir(os.path.join(poet_path, 'ur')))
                        en_files = len(os.listdir(os.path.join(poet_path, 'en')))
                        print(f"✅ Urdu files: {ur_files}, English files: {en_files}")
                        return dataset_dir
                    else:
                        print(f"❌ Invalid structure - missing 'ur' or 'en' folders")
                        return None
                else:
                    print("❌ No poet directories found")
                    return None
            else:
                print("❌ Could not find dataset directory with poets")
                return None
                
    except Exception as e:
        print(f"❌ Error extracting zip: {e}")
        return None

def create_fixed_data_loader():
    """
    Create a data loader that works with the extracted dataset
    """
    code = '''
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
        text = re.sub(r'\\s+', ' ', text)
        
        # Remove unwanted punctuation but keep essential ones
        text = re.sub(r'[^\\u0600-\\u06FF\\u0750-\\u077F\\u08A0-\\u08FF\\s۔،؍؎؏؟!]', '', text)
        
        return text.strip()
    
    @staticmethod
    def clean_roman(text: str) -> str:
        """
        Clean and normalize Roman Urdu text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        # Keep only alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-ZāīūĀĪŪñṇṛṭḍṣġḥḳẓẕ\\s\\'\\-\\.]', '', text)
        
        return text.strip()
    
    @staticmethod
    def add_special_tokens(text: str, is_target: bool = False) -> str:
        """
        Add special tokens for sequence processing
        """
        if is_target:
            return f"<sos> {text} <eos>"
        return text

def load_dataset(data_path):
    """
    Load Urdu-Roman Urdu parallel corpus from the extracted dataset
    """
    print(f"🔍 Loading dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset path does not exist: {data_path}")
        return [], []
    
    urdu_texts = []
    roman_texts = []
    
    # Get all poet directories
    poets = [d for d in os.listdir(data_path) 
             if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('.')]
    
    print(f"📚 Found {len(poets)} poets in dataset")
    
    if len(poets) == 0:
        print("❌ No poet directories found!")
        return [], []
    
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
            
            print(f"  📖 {poet}: {len(common_files)} common files")
            
            for file_name in common_files:
                try:
                    # Read Urdu text
                    urdu_file_path = os.path.join(urdu_path, file_name)
                    with open(urdu_file_path, 'r', encoding='utf-8') as f:
                        urdu_content = f.read().strip()
                    
                    # Read Roman Urdu text
                    roman_file_path = os.path.join(english_path, file_name)
                    with open(roman_file_path, 'r', encoding='utf-8') as f:
                        roman_content = f.read().strip()
                    
                    # Split into lines and pair them
                    urdu_lines = [line.strip() for line in urdu_content.split('\\n') if line.strip()]
                    roman_lines = [line.strip() for line in roman_content.split('\\n') if line.strip()]
                    
                    # Ensure same number of lines
                    min_lines = min(len(urdu_lines), len(roman_lines))
                    for i in range(min_lines):
                        if urdu_lines[i] and roman_lines[i]:
                            urdu_texts.append(urdu_lines[i])
                            roman_texts.append(roman_lines[i])
                            
                except Exception as e:
                    print(f"  ⚠️ Error processing {poet}/{file_name}: {e}")
                    continue
        else:
            print(f"  ❌ {poet}: Missing ur or en directory")
    
    print(f"\\n📊 Dataset loaded:")
    print(f"  Total pairs: {len(urdu_texts)}")
    
    if len(urdu_texts) > 0:
        print(f"  Sample Urdu: {urdu_texts[0]}")
        print(f"  Sample Roman: {roman_texts[0]}")
    
    return urdu_texts, roman_texts

def clean_and_split_data(urdu_texts: List[str], roman_texts: List[str], 
                        train_ratio: float = 0.5, val_ratio: float = 0.25, test_ratio: float = 0.25):
    """
    Clean data and split into train/val/test sets
    """
    if len(urdu_texts) == 0:
        print("❌ No data to clean and split!")
        return [], [], []
    
    cleaner = TextCleaner()
    
    print("🧹 Cleaning texts...")
    cleaned_urdu = [cleaner.clean_urdu(text) for text in tqdm(urdu_texts, desc="Cleaning Urdu")]
    cleaned_roman = [cleaner.clean_roman(text) for text in tqdm(roman_texts, desc="Cleaning Roman")]
    
    # Filter out empty pairs and very short/long sequences
    valid_pairs = []
    for u, r in zip(cleaned_urdu, cleaned_roman):
        if u and r and 2 <= len(u.split()) <= 50 and 2 <= len(r.split()) <= 50:
            valid_pairs.append((u, r))
    
    print(f"✅ After cleaning and filtering: {len(valid_pairs)} valid pairs")
    
    if len(valid_pairs) == 0:
        print("❌ No valid pairs after filtering!")
        return [], [], []
    
    # Shuffle data
    random.shuffle(valid_pairs)
    
    # Split data
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
'''
    
    # Write the fixed data loader
    with open('/content/data_loader_fixed.py', 'w') as f:
        f.write(code)
    
    print("✅ Created fixed data loader at /content/data_loader_fixed.py")

def main():
    """
    Main function to fix dataset issues in Google Colab
    """
    print("🚀 FIXING DATASET FOR GOOGLE COLAB")
    print("=" * 50)
    
    # Step 1: Extract the dataset
    dataset_path = extract_dataset_zip()
    
    if dataset_path:
        print(f"\\n🎉 SUCCESS! Dataset extracted and ready!")
        print(f"📍 Dataset path: {dataset_path}")
        
        # Step 2: Create fixed data loader
        create_fixed_data_loader()
        
        # Step 3: Provide usage instructions
        print(f"\\n📝 USAGE INSTRUCTIONS:")
        print(f"1. Use this path in your code:")
        print(f"   dataset_path = '{dataset_path}'")
        print(f"")
        print(f"2. Replace the import in your training code:")
        print(f"   # OLD: from data_loader import load_dataset, clean_and_split_data")
        print(f"   # NEW: from data_loader_fixed import load_dataset, clean_and_split_data")
        print(f"")
        print(f"3. Or run this in your notebook:")
        print(f"   exec(open('/content/data_loader_fixed.py').read())")
        
        return dataset_path
    else:
        print(f"\\n❌ FAILED! Could not extract or find dataset")
        return None

if __name__ == "__main__":
    main()

