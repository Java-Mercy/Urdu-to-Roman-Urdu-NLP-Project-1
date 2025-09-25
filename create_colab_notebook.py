#!/usr/bin/env python3
"""
Script to create a complete Google Colab notebook with all code embedded
"""

import json

def create_colab_notebook():
    """
    Create a comprehensive Jupyter notebook for Google Colab
    """
    
    notebook = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    # Add cells
    cells = [
        # Title cell
        {
            "cell_type": "markdown",
            "metadata": {"id": "title"},
            "source": [
                "# Neural Machine Translation: Urdu to Roman Urdu\n",
                "## BiLSTM Encoder-Decoder Architecture\n",
                "\n",
                "**Assignment**: Project1 - Neural Machine Translation (15 Abs)  \n",
                "**Objective**: Build a sequence-to-sequence model using BiLSTM encoder-decoder to translate Urdu text into Roman Urdu transliteration.\n",
                "\n",
                "**Architecture**:\n",
                "- Encoder: 2-layer Bidirectional LSTM\n",
                "- Decoder: 4-layer LSTM\n",
                "- Custom BPE Tokenization (implemented from scratch)\n",
                "\n",
                "**Dataset**: urdu_ghazals_rekhta - Classical Urdu poetry with Roman transliterations"
            ]
        },
        
        # Setup cell
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "setup"},
            "outputs": [],
            "source": [
                "# Install required packages\n",
                "%pip install torch torchtext nltk sacrebleu editdistance streamlit\n",
                "%pip install matplotlib seaborn tqdm pandas numpy"
            ]
        },
        
        # Imports cell
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "imports"},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "import torch.nn.functional as F\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from torch.nn.utils.rnn import pad_sequence\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from tqdm import tqdm\n",
                "import re\n",
                "import os\n",
                "import json\n",
                "import pickle\n",
                "from collections import Counter, defaultdict\n",
                "import random\n",
                "from typing import List, Tuple, Dict, Any\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set device\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f\"Using device: {device}\")\n",
                "\n",
                "# Set random seeds for reproducibility\n",
                "torch.manual_seed(42)\n",
                "np.random.seed(42)\n",
                "random.seed(42)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed(42)"
            ]
        },
        
        # Dataset download cell
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "dataset"},
            "outputs": [],
            "source": [
                "# Clone the dataset repository\n",
                "!git clone https://github.com/amir9ume/urdu_ghazals_rekhta.git\n",
                "\n",
                "# Set dataset path\n",
                "if os.path.exists('/content/urdu_ghazals_rekhta/dataset'):\n",
                "    dataset_path = '/content/urdu_ghazals_rekhta/dataset'\n",
                "elif os.path.exists('urdu_ghazals_rekhta/dataset'):\n",
                "    dataset_path = 'urdu_ghazals_rekhta/dataset'\n",
                "else:\n",
                "    dataset_path = 'dataset/dataset'\n",
                "\n",
                "print(f\"Dataset path: {dataset_path}\")"
            ]
        }
    ]
    
    # Read and add implementation files as code cells
    file_contents = {}
    files_to_include = [
        'data_loader.py',
        'bpe_tokenizer.py', 
        'model.py',
        'dataset.py',
        'training.py',
        'evaluation.py'
    ]
    
    for filename in files_to_include:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            cells.append({
                "cell_type": "markdown",
                "metadata": {"id": f"header_{filename.replace('.py', '')}"},
                "source": [f"## {filename.replace('.py', '').replace('_', ' ').title()}"]
            })
            
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": filename.replace('.py', '')},
                "outputs": [],
                "source": [content]
            })
            
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
    
    # Add training execution cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "execution_header"},
        "source": ["## Training Execution"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "execution"},
        "outputs": [],
        "source": [
            "# Main training execution\n",
            "\n",
            "# Set up experiment configuration\n",
            "config = {\n",
            "    'seed': 42,\n",
            "    'embed_dim': 256,\n",
            "    'hidden_dim': 512,\n",
            "    'encoder_layers': 2,\n",
            "    'decoder_layers': 4,\n",
            "    'dropout': 0.3,\n",
            "    'learning_rate': 1e-3,\n",
            "    'batch_size': 32,  # Reduced for Colab free tier\n",
            "    'src_vocab_size': 8000,\n",
            "    'tgt_vocab_size': 8000,\n",
            "    'num_epochs': 10  # Reduced for Colab free tier\n",
            "}\n",
            "\n",
            "print(\"Starting Urdu to Roman Urdu NMT Training...\")\n",
            "print(f\"Configuration: {config}\")\n",
            "\n",
            "# 1. Load and preprocess data\n",
            "print(\"\\n1. Loading and preprocessing data...\")\n",
            "urdu_texts, roman_texts = load_dataset(dataset_path)\n",
            "train_pairs, val_pairs, test_pairs = clean_and_split_data(\n",
            "    urdu_texts, roman_texts,\n",
            "    train_ratio=0.5, val_ratio=0.25, test_ratio=0.25\n",
            ")\n",
            "\n",
            "# 2. Create tokenizers\n",
            "print(\"\\n2. Training tokenizers...\")\n",
            "src_tokenizer, tgt_tokenizer = create_tokenizers(\n",
            "    train_pairs,\n",
            "    src_vocab_size=config['src_vocab_size'],\n",
            "    tgt_vocab_size=config['tgt_vocab_size']\n",
            ")\n",
            "\n",
            "# 3. Create data loaders\n",
            "print(\"\\n3. Creating data loaders...\")\n",
            "train_loader, val_loader, test_loader = create_data_loaders(\n",
            "    train_pairs, val_pairs, test_pairs,\n",
            "    src_tokenizer, tgt_tokenizer,\n",
            "    batch_size=config['batch_size']\n",
            ")\n",
            "\n",
            "# 4. Create model\n",
            "print(\"\\n4. Creating model...\")\n",
            "model = Seq2SeqModel(\n",
            "    src_vocab_size=src_tokenizer.get_vocab_size(),\n",
            "    tgt_vocab_size=tgt_tokenizer.get_vocab_size(),\n",
            "    embed_dim=config['embed_dim'],\n",
            "    hidden_dim=config['hidden_dim'],\n",
            "    encoder_layers=config['encoder_layers'],\n",
            "    decoder_layers=config['decoder_layers'],\n",
            "    dropout=config['dropout']\n",
            ")\n",
            "\n",
            "print(f\"Model parameters: {sum(p.numel() for p in model.parameters()):,}\")\n",
            "\n",
            "# 5. Train model\n",
            "print(\"\\n5. Starting training...\")\n",
            "trainer = Trainer(\n",
            "    model=model,\n",
            "    train_loader=train_loader,\n",
            "    val_loader=val_loader,\n",
            "    src_tokenizer=src_tokenizer,\n",
            "    tgt_tokenizer=tgt_tokenizer,\n",
            "    lr=config['learning_rate'],\n",
            "    device=device\n",
            ")\n",
            "\n",
            "train_losses, val_losses = trainer.train(\n",
            "    num_epochs=config['num_epochs'],\n",
            "    save_path='best_model.pth'\n",
            ")\n",
            "\n",
            "# 6. Plot training curves\n",
            "trainer.plot_training_curves('training_curves.png')\n",
            "\n",
            "# 7. Evaluate model\n",
            "print(\"\\n6. Evaluating model...\")\n",
            "evaluator = Evaluator(model, test_loader, src_tokenizer, tgt_tokenizer, device)\n",
            "results = evaluator.evaluate(num_samples=200)  # Reduced for speed\n",
            "\n",
            "print(\"\\n=\" * 60)\n",
            "print(\"TRAINING COMPLETED!\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"Final Results:\")\n",
            "print(f\"  BLEU Score: {results['bleu']:.4f}\")\n",
            "print(f\"  Character Error Rate: {results['cer']:.4f}\")\n",
            "print(f\"  Perplexity: {results['perplexity']:.4f}\")\n",
            "\n",
            "# 8. Test with sample translations\n",
            "print(\"\\n7. Sample translations:\")\n",
            "sample_texts = [\n",
            "    \"محبت میں نہیں ہے فرق جینے اور مرنے کا\",\n",
            "    \"دل ہی تو ہے نہ سنگ و خشت درد سے بھر نہ آئے کیوں\",\n",
            "    \"ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے\"\n",
            "]\n",
            "\n",
            "for text in sample_texts:\n",
            "    translation = translate_text(model, text, src_tokenizer, tgt_tokenizer, device)\n",
            "    print(f\"Urdu: {text}\")\n",
            "    print(f\"Roman: {translation}\")\n",
            "    print()"
        ]
    })
    
    # Add notebook cells
    notebook["cells"] = cells
    
    # Save notebook
    with open('complete_urdu_roman_nmt.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("Complete notebook created: complete_urdu_roman_nmt.ipynb")
    print("This notebook can be uploaded directly to Google Colab!")

if __name__ == "__main__":
    create_colab_notebook()
