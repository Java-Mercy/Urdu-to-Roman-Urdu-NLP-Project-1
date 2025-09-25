import torch
import numpy as np
from tqdm import tqdm
import editdistance
from collections import Counter
import math
import re

class Evaluator:
    """
    Evaluation metrics for translation quality
    """
    
    def __init__(self, model, test_loader, src_tokenizer, tgt_tokenizer, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
    
    def calculate_bleu(self, references, hypotheses, n=4):
        """
        Calculate BLEU score
        """
        def get_ngrams(tokens, n):
            if len(tokens) < n:
                return Counter()
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(tuple(tokens[i:i+n]))
            return Counter(ngrams)
        
        def calculate_precision(ref_tokens, hyp_tokens, n):
            ref_ngrams = get_ngrams(ref_tokens, n)
            hyp_ngrams = get_ngrams(hyp_tokens, n)
            
            if not hyp_ngrams:
                return 0.0
            
            matches = 0
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            return matches / sum(hyp_ngrams.values())
        
        # Calculate precision for each n-gram order
        precisions = []
        for i in range(1, n + 1):
            precision_scores = []
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                precision = calculate_precision(ref_tokens, hyp_tokens, i)
                precision_scores.append(precision)
            precisions.append(np.mean(precision_scores))
        
        # Calculate brevity penalty
        ref_lengths = [len(ref.split()) for ref in references]
        hyp_lengths = [len(hyp.split()) for hyp in hypotheses]
        
        ref_len = sum(ref_lengths)
        hyp_len = sum(hyp_lengths)
        
        if hyp_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
        
        # Calculate BLEU score
        if any(p == 0 for p in precisions):
            return 0.0
        
        geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        bleu = bp * geometric_mean
        
        return bleu
    
    def calculate_cer(self, references, hypotheses):
        """
        Calculate Character Error Rate
        """
        total_chars = 0
        total_errors = 0
        
        for ref, hyp in zip(references, hypotheses):
            ref_chars = list(ref.replace(' ', ''))
            hyp_chars = list(hyp.replace(' ', ''))
            
            total_chars += len(ref_chars)
            total_errors += editdistance.eval(ref_chars, hyp_chars)
        
        return total_errors / total_chars if total_chars > 0 else 1.0
    
    def calculate_perplexity(self):
        """
        Calculate perplexity on test set
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Calculating perplexity'):
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_lengths = batch['src_lengths'].to(self.device)
                
                outputs = self.model(src, tgt, src_lengths, teacher_forcing_ratio=0)
                
                outputs = outputs[:, 1:].contiguous().view(-1, outputs.size(-1))
                targets = tgt[:, 1:].contiguous().view(-1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                total_tokens += targets.ne(0).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        perplexity = np.exp(total_loss * len(self.test_loader) / total_tokens)
        
        return perplexity
    
    def generate_translations(self, num_samples=None):
        """
        Generate translations for the test set
        """
        self.model.eval()
        references = []
        hypotheses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Generating translations')):
                if num_samples and batch_idx * self.test_loader.batch_size >= num_samples:
                    break
                
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # Generate translations
                generated = self.model.inference(src, self.src_tokenizer, self.tgt_tokenizer)
                
                # Decode sequences
                for i in range(src.size(0)):
                    # Reference (remove SOS and EOS tokens)
                    ref_tokens = tgt[i].cpu().numpy()
                    ref_tokens = ref_tokens[1:]  # Remove SOS
                    eos_idx = np.where(ref_tokens == self.tgt_tokenizer.vocab['<eos>'])[0]
                    if len(eos_idx) > 0:
                        ref_tokens = ref_tokens[:eos_idx[0]]
                    
                    ref_text = self.tgt_tokenizer.decode(ref_tokens.tolist())
                    
                    # Hypothesis
                    if generated.size(1) > 0:
                        hyp_tokens = generated[i].cpu().numpy()
                        eos_idx = np.where(hyp_tokens == self.tgt_tokenizer.vocab['<eos>'])[0]
                        if len(eos_idx) > 0:
                            hyp_tokens = hyp_tokens[:eos_idx[0]]
                        hyp_text = self.tgt_tokenizer.decode(hyp_tokens.tolist())
                    else:
                        hyp_text = ""
                    
                    references.append(ref_text)
                    hypotheses.append(hyp_text)
        
        return references, hypotheses
    
    def evaluate(self, num_samples=None):
        """
        Comprehensive evaluation
        """
        print("Starting evaluation...")
        
        # Generate translations
        references, hypotheses = self.generate_translations(num_samples)
        
        # Calculate metrics
        bleu_score = self.calculate_bleu(references, hypotheses)
        cer = self.calculate_cer(references, hypotheses)
        perplexity = self.calculate_perplexity()
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"  BLEU Score: {bleu_score:.4f}")
        print(f"  Character Error Rate: {cer:.4f}")
        print(f"  Perplexity: {perplexity:.4f}")
        
        # Show some examples
        print(f"\nSample Translations:")
        for i in range(min(5, len(references))):
            print(f"\n{i+1}.")
            print(f"Reference: {references[i]}")
            print(f"Generated: {hypotheses[i]}")
        
        return {
            'bleu': bleu_score,
            'cer': cer,
            'perplexity': perplexity,
            'references': references,
            'hypotheses': hypotheses
        }

def translate_text(model, text, src_tokenizer, tgt_tokenizer, device='cuda', max_length=50):
    """
    Translate a single text using the trained model
    """
    model.eval()
    
    # Tokenize input
    src_tokens = src_tokenizer.encode(text)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    
    # Generate translation
    with torch.no_grad():
        generated = model.inference(src_tensor, src_tokenizer, tgt_tokenizer, max_length)
    
    # Decode output
    if generated.size(1) > 0:
        output_tokens = generated[0].cpu().numpy()
        eos_idx = np.where(output_tokens == tgt_tokenizer.vocab['<eos>'])[0]
        if len(eos_idx) > 0:
            output_tokens = output_tokens[:eos_idx[0]]
        translation = tgt_tokenizer.decode(output_tokens.tolist())
    else:
        translation = ""
    
    return translation
