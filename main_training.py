#!/usr/bin/env python3
"""
Main training script for Urdu to Roman Urdu Neural Machine Translation
Assignment: Project1 - Neural Machine Translation (15 Abs)
"""

import torch
import numpy as np
import random
import json
import os
from datetime import datetime

# Import our modules
from data_loader import load_dataset, clean_and_split_data
from bpe_tokenizer import create_tokenizers
from model import Seq2SeqModel
from dataset import create_data_loaders
from training import Trainer
from evaluation import Evaluator, translate_text

# Set random seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def run_experiment(config, experiment_name):
    """
    Run a single experiment with given configuration
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Set seeds
    set_seeds(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and split data
    print("\n1. Loading and preprocessing data...")
    urdu_texts, roman_texts = load_dataset('dataset/dataset')
    train_pairs, val_pairs, test_pairs = clean_and_split_data(
        urdu_texts, roman_texts,
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
    )
    
    # Create tokenizers
    print("\n2. Training tokenizers...")
    src_tokenizer, tgt_tokenizer = create_tokenizers(
        train_pairs, 
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size']
    )
    
    # Save tokenizers
    os.makedirs(f'experiments/{experiment_name}', exist_ok=True)
    src_tokenizer.save(f'experiments/{experiment_name}/src_tokenizer.pkl')
    tgt_tokenizer.save(f'experiments/{experiment_name}/tgt_tokenizer.pkl')
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_pairs, val_pairs, test_pairs,
        src_tokenizer, tgt_tokenizer,
        batch_size=config['batch_size']
    )
    
    # Create model
    print("\n4. Creating model...")
    model = Seq2SeqModel(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        dropout=config['dropout']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\n5. Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        lr=config['learning_rate'],
        device=device
    )
    
    # Train model
    model_path = f'experiments/{experiment_name}/best_model.pth'
    train_losses, val_losses = trainer.train(
        num_epochs=config['num_epochs'],
        save_path=model_path
    )
    
    # Plot training curves
    trainer.plot_training_curves(f'experiments/{experiment_name}/training_curves.png')
    
    # Evaluate model
    print("\n6. Evaluating model...")
    evaluator = Evaluator(model, test_loader, src_tokenizer, tgt_tokenizer, device)
    results = evaluator.evaluate(num_samples=500)  # Evaluate on subset for speed
    
    # Save results
    experiment_results = {
        'config': config,
        'experiment_name': experiment_name,
        'results': results,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }
    
    with open(f'experiments/{experiment_name}/results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    # Test with sample translations
    print("\n7. Sample translations:")
    sample_texts = [
        "محبت میں نہیں ہے فرق جینے اور مرنے کا",
        "دل ہی تو ہے نہ سنگ و خشت درد سے بھر نہ آئے کیوں",
        "ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے"
    ]
    
    for text in sample_texts:
        translation = translate_text(model, text, src_tokenizer, tgt_tokenizer, device)
        print(f"Urdu: {text}")
        print(f"Roman: {translation}")
        print()
    
    return experiment_results

def main():
    """
    Main function to run all experiments
    """
    print("Urdu to Roman Urdu Neural Machine Translation")
    print("BiLSTM Encoder-Decoder Architecture")
    print(f"Started at: {datetime.now()}")
    
    # Experiment configurations
    experiments = {
        'baseline': {
            'seed': 42,
            'embed_dim': 256,
            'hidden_dim': 512,
            'encoder_layers': 2,
            'decoder_layers': 4,
            'dropout': 0.3,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'src_vocab_size': 8000,
            'tgt_vocab_size': 8000,
            'num_epochs': 20
        },
        
        'exp1_embedding_dim': {
            'seed': 42,
            'embed_dim': 128,  # Changed from 256
            'hidden_dim': 512,
            'encoder_layers': 2,
            'decoder_layers': 4,
            'dropout': 0.3,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'src_vocab_size': 8000,
            'tgt_vocab_size': 8000,
            'num_epochs': 20
        },
        
        'exp2_hidden_dim': {
            'seed': 42,
            'embed_dim': 256,
            'hidden_dim': 256,  # Changed from 512
            'encoder_layers': 2,
            'decoder_layers': 4,
            'dropout': 0.3,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'src_vocab_size': 8000,
            'tgt_vocab_size': 8000,
            'num_epochs': 20
        },
        
        'exp3_dropout': {
            'seed': 42,
            'embed_dim': 256,
            'hidden_dim': 512,
            'encoder_layers': 2,
            'decoder_layers': 4,
            'dropout': 0.5,  # Changed from 0.3
            'learning_rate': 1e-3,
            'batch_size': 64,
            'src_vocab_size': 8000,
            'tgt_vocab_size': 8000,
            'num_epochs': 20
        },
        
        'exp4_learning_rate': {
            'seed': 42,
            'embed_dim': 256,
            'hidden_dim': 512,
            'encoder_layers': 2,
            'decoder_layers': 4,
            'dropout': 0.3,
            'learning_rate': 5e-4,  # Changed from 1e-3
            'batch_size': 64,
            'src_vocab_size': 8000,
            'tgt_vocab_size': 8000,
            'num_epochs': 20
        }
    }
    
    # Run experiments
    all_results = {}
    
    for exp_name, config in experiments.items():
        try:
            results = run_experiment(config, exp_name)
            all_results[exp_name] = results
            
            print(f"\nExperiment {exp_name} completed!")
            print(f"BLEU: {results['results']['bleu']:.4f}")
            print(f"CER: {results['results']['cer']:.4f}")
            print(f"Perplexity: {results['results']['perplexity']:.4f}")
            
        except Exception as e:
            print(f"Error in experiment {exp_name}: {str(e)}")
            continue
    
    # Compare results
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Experiment':<20} {'BLEU':<8} {'CER':<8} {'Perplexity':<12} {'Parameters':<12}")
    print("-" * 80)
    
    for exp_name, results in all_results.items():
        if 'results' in results:
            bleu = results['results']['bleu']
            cer = results['results']['cer']
            ppl = results['results']['perplexity']
            params = results['model_parameters']
            print(f"{exp_name:<20} {bleu:<8.4f} {cer:<8.4f} {ppl:<12.2f} {params:<12,}")
    
    # Save combined results
    with open('experiments/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll experiments completed at: {datetime.now()}")
    print("Results saved in 'experiments/' directory")

if __name__ == "__main__":
    main()
