import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from typing import List, Tuple, Dict, Iterable

from data.english_sentences import en_sentences
from data.german_sentences import de_sentences

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data, src_tokenizer, tgt_tokenizer, 
                src_vocab, tgt_vocab, max_seq_length=100):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_seq_length = max_seq_length
        
        # Special tokens
        self.PAD_IDX = 0
        self.BOS_IDX = 2
        self.EOS_IDX = 3
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]
        
        # Tokenize and convert to indices
        src_tokens = self.src_tokenizer(src_text)
        tgt_tokens = self.tgt_tokenizer(tgt_text)
        
        # Truncate if necessary
        if len(src_tokens) > self.max_seq_length - 2:
            src_tokens = src_tokens[:self.max_seq_length - 2]
        if len(tgt_tokens) > self.max_seq_length - 2:
            tgt_tokens = tgt_tokens[:self.max_seq_length - 2]
        
        # Add BOS and EOS tokens
        src_indices = [self.BOS_IDX] + [self.src_vocab.get(token, 1) for token in src_tokens] + [self.EOS_IDX]
        tgt_indices = [self.BOS_IDX] + [self.tgt_vocab.get(token, 1) for token in tgt_tokens] + [self.EOS_IDX]
        
        # Convert to tensors
        src_tensor = torch.LongTensor(src_indices)
        tgt_tensor = torch.LongTensor(tgt_indices)
        
        return src_tensor, tgt_tensor

def load_text_data(en_file, de_file):
    en_path = "data/" + en_file
    de_path = "data/" + de_file
    
    with open(en_path, 'r', encoding='utf-8') as f:
        en_sentences = [line.strip() for line in f.readlines()]
    
    with open(de_path, 'r', encoding='utf-8') as f:
        de_sentences = [line.strip() for line in f.readlines()]
    
    # Filter out empty lines and ensure parallel data
    filtered_pairs = [(en, de) for en, de in zip(en_sentences, de_sentences) 
                     if en.strip() and de.strip()]
    
    en_sentences = [pair[0] for pair in filtered_pairs]
    de_sentences = [pair[1] for pair in filtered_pairs]
    
    print(f"Loaded {len(en_sentences)} sentence pairs")
    return en_sentences, de_sentences

def load_tokenizers():
    try:
        spacy_en = spacy.load("en_core_web_sm")
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        print("Downloading spacy models")
        import os
        os.system("python -m spacy download en_core_web_sm")
        os.system("python -m spacy download de_core_news_sm")
        spacy_en = spacy.load("en_core_web_sm")
        spacy_de = spacy.load("de_core_news_sm")
    
    return spacy_en, spacy_de

def tokenize_en(text, spacy_en):
    return [token.text.lower() for token in spacy_en.tokenizer(text)]

def tokenize_de(text, spacy_de):
    return [token.text.lower() for token in spacy_de.tokenizer(text)]

def build_vocab(sentences, tokenizer, min_freq=2):
    word_counts = {}
    for sentence in sentences:
        tokens = tokenizer(sentence)
        for token in tokens:
            token = token.lower()
            word_counts[token] = word_counts.get(token, 0) + 1
    
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    idx = 4
    
    for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

def load_multi30k_dataset():
    # Load training data
    en_sentences, de_sentences = load_text_data(
        "Customer-sample-English-German-Training-en.txt",
        "Customer-sample-English-German-Training-de.txt"
    )
    
# Load tokenizers
    spacy_en, spacy_de = load_tokenizers()
    en_tokenizer = lambda text: tokenize_en(text, spacy_en)
    de_tokenizer = lambda text: tokenize_de(text, spacy_de)
    
    # Build vocabularies
    en_vocab = build_vocab(en_sentences, en_tokenizer, min_freq=1)
    de_vocab = build_vocab(de_sentences, de_tokenizer, min_freq=1)
    
    # Split ratios: 80% training, 10% validation, 10% testing
    total_size = len(en_sentences)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    
    # Create splits
    train_en = en_sentences[:train_size]
    train_de = de_sentences[:train_size]
    
    val_en = en_sentences[train_size:train_size+val_size]
    val_de = de_sentences[train_size:train_size+val_size]
    
    test_en = en_sentences[train_size+val_size:]
    test_de = de_sentences[train_size+val_size:]
    
    print(f"Total pairs: {total_size}")
    print(f"Training pairs: {len(train_en)}")
    print(f"Validation pairs: {len(val_en)}")
    print(f"Test pairs: {len(test_en)}")
    print(f"English vocabulary size: {len(en_vocab)}")
    print(f"German vocabulary size: {len(de_vocab)}")
    
    # Create datasets
    train_dataset = TranslationDataset(
        train_en, train_de,
        en_tokenizer, de_tokenizer, en_vocab, de_vocab
    )
    
    val_dataset = TranslationDataset(
        val_en, val_de,
        en_tokenizer, de_tokenizer, en_vocab, de_vocab
    )
    
    test_dataset = TranslationDataset(
        test_en, test_de,
        en_tokenizer, de_tokenizer, en_vocab, de_vocab
    )
    
    return train_dataset, val_dataset, test_dataset, en_vocab, de_vocab

def collate_batch(batch, pad_idx=0):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=True)
    
    return src_batch, tgt_batch