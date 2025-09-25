import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os

class Trainer:
    """
    Training class for the Seq2Seq model
    """
    
    def __init__(self, model, train_loader, val_loader, src_tokenizer, tgt_tokenizer, 
                 lr=1e-3, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        
    def train_epoch(self, epoch, teacher_forcing_ratio=0.5):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_lengths = batch['src_lengths'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(src, tgt, src_lengths, teacher_forcing_ratio)
            
            # Calculate loss (ignore first token which is SOS)
            outputs = outputs[:, 1:].contiguous().view(-1, outputs.size(-1))
            targets = tgt[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_tokens += targets.ne(0).sum().item()  # Count non-padding tokens
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        perplexity = np.exp(total_loss * len(self.train_loader) / total_tokens)
        
        return avg_loss, perplexity
    
    def validate(self):
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_lengths = batch['src_lengths'].to(self.device)
                
                outputs = self.model(src, tgt, src_lengths, teacher_forcing_ratio=0)
                
                outputs = outputs[:, 1:].contiguous().view(-1, outputs.size(-1))
                targets = tgt[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                total_tokens += targets.ne(0).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = np.exp(total_loss * len(self.val_loader) / total_tokens)
        
        return avg_loss, perplexity
    
    def train(self, num_epochs, save_path='best_model.pth'):
        """
        Full training loop
        """
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss, train_perplexity = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_perplexity = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_perplexities.append(train_perplexity)
            self.val_perplexities.append(val_perplexity)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}')
            print(f'  Time: {epoch_time:.2f}s, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_perplexities': self.train_perplexities,
                    'val_perplexities': self.val_perplexities,
                    'best_val_loss': best_val_loss
                }, save_path)
                print(f'  New best model saved! Val Loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= max_patience:
                print(f'Early stopping after {epoch} epochs')
                break
                
            print('-' * 60)
        
        print('Training completed!')
        return self.train_losses, self.val_losses
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """
        Plot training and validation curves
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Perplexity curves
        ax2.plot(self.train_perplexities, label='Train Perplexity', color='blue')
        ax2.plot(self.val_perplexities, label='Validation Perplexity', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Training and Validation Perplexity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def load_model(model_class, model_path, src_vocab_size, tgt_vocab_size, device='cuda', **model_kwargs):
    """
    Load a trained model
    """
    model = model_class(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, **model_kwargs)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint
