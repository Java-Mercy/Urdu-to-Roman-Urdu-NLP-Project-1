import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import pickle

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer implemented from scratch
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_freqs = Counter()
        self.vocab = {}
        self.merges = []
        self.special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        
    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Get frequency of consecutive symbol pairs
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """
        Merge the most frequent pair in the vocabulary
        """
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def train(self, texts: List[str]):
        """
        Train BPE on the given texts
        """
        print("Training BPE tokenizer...")
        
        # Initialize word frequencies
        for text in texts:
            words = text.split()
            for word in words:
                self.word_freqs[word] += 1
        
        # Initialize vocabulary with character-level splits
        vocab = {}
        for word, freq in self.word_freqs.items():
            # Split word into characters and add end-of-word token
            vocab[' '.join(list(word)) + ' </w>'] = freq
        
        # Add special tokens to vocabulary
        for token in self.special_tokens:
            vocab[token] = 1
        
        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.special_tokens)
        
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            if (i + 1) % 1000 == 0:
                print(f"Merged {i + 1}/{num_merges} pairs")
        
        # Create final vocabulary
        self.vocab = {}
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
        
        for word in vocab:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        print(f"BPE training completed. Vocabulary size: {len(self.vocab)}")
    
    def _get_word_tokens(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned BPE merges
        """
        if word in self.vocab:
            return [word]
        
        word = ' '.join(list(word)) + ' </w>'
        pairs = self._get_word_pairs(word)
        
        if not pairs:
            return [word]
        
        while True:
            bigram = min(pairs, key=lambda pair: self.merges.index(pair) if pair in self.merges else float('inf'))
            if bigram not in self.merges:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_word_pairs(word)
        
        return word
    
    def _get_word_pairs(self, word) -> Set[Tuple[str, str]]:
        """
        Get all pairs from a word
        """
        if isinstance(word, str):
            word = word.split()
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs
        """
        tokens = []
        words = text.split()
        
        for word in words:
            word_tokens = self._get_word_tokens(word)
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab['<unk>'])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        """
        # Create reverse vocabulary
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)
        
        # Join tokens and clean up
        text = ' '.join(tokens)
        text = text.replace('</w>', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def save(self, filepath: str):
        """
        Save tokenizer to file
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'merges': self.merges,
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }, f)
    
    def load(self, filepath: str):
        """
        Load tokenizer from file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.vocab_size = data['vocab_size']
            self.special_tokens = data['special_tokens']
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)

def create_tokenizers(train_pairs: List[Tuple[str, str]], 
                     src_vocab_size: int = 8000, 
                     tgt_vocab_size: int = 8000) -> Tuple[BPETokenizer, BPETokenizer]:
    """
    Create and train source and target tokenizers
    """
    src_texts = [pair[0] for pair in train_pairs]
    tgt_texts = [pair[1] for pair in train_pairs]
    
    # Create tokenizers
    src_tokenizer = BPETokenizer(vocab_size=src_vocab_size)
    tgt_tokenizer = BPETokenizer(vocab_size=tgt_vocab_size)
    
    # Train tokenizers
    print("Training source (Urdu) tokenizer...")
    src_tokenizer.train(src_texts)
    
    print("Training target (Roman Urdu) tokenizer...")
    tgt_tokenizer.train(tgt_texts)
    
    return src_tokenizer, tgt_tokenizer
