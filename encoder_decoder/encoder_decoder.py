# import numpy as np
# import pandas as pd

import torch
import torch.nn as nn
import math
import copy

class Transformer(nn.Module):
    """
    Full Transformer model for seq2seq translations
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model = 512,
                 nhead = 6,
                 num_encoder_layer = 6,
                 num_decoder_layer = 6,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                max_seq_length = 5000 ):
        super().__init__()

        # Embedding for source and target sequences
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

         # Positional encoding to give model information about token positions
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)

        # Encoder and Decoder
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = Encoder(encoder_layer, num_encoder_layer)

        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = Decoder(decoder_layer, num_decoder_layer)

        # Final linear layer to project to vocabulary size
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters with Xavier/Glorot initialization
        self._init_parameters()

        self.d_model = d_model

    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        """
        Forward pass of the transformer
        
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len]
            src_mask: Mask for source sequence to avoid attending to padding
            tgt_mask: Mask for target sequence (prevents attending to future tokens)
            src_padding_mask: Mask for source padding tokens
            tgt_padding_mask: Mask for target padding tokens
            
        Returns:
            Output tensor of shape [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Embed and add positional encoding to source sequence
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        
        # Embed and add positional encoding to target sequence
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        
        # Pass through encoder
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)
        
        # Pass through decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, 
                             memory_mask=src_mask,
                             tgt_key_padding_mask=tgt_padding_mask,
                             memory_key_padding_mask=src_padding_mask)
        
        # Project to vocabulary size
        output = self.generator(output)
        return output
    

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Calculate sine and cosine positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (won't be updated during backprop)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in the paper
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Linear projections for Q, K, V, and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Implements multi-head attention
        
        Args:
            query: Query tensor [batch_size, query_len, d_model]
            key: Key tensor [batch_size, key_len, d_model]
            value: Value tensor [batch_size, value_len, d_model]
            attn_mask: Optional mask to prevent attention to certain positions
            key_padding_mask: Optional mask for padding tokens
            
        Returns:
            Output tensor after multi-head attention [batch_size, query_len, d_model]
        """
        batch_size = query.size(0)
        
        # Linear projections and split into multiple heads
        q = self.q_linear(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply masks if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attn_output)
        
        return output
    

class Encoder(nn.Module):
    """
    Transformer Encoder consisting of N identical layers
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm if norm is not None else nn.LayerNorm(encoder_layer.size)
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Pass the input through each encoder layer in turn
        
        Args:
            src: Source sequence [batch_size, src_seq_len, d_model]
            mask: Optional mask for source sequence
            src_key_padding_mask: Optional mask for padding tokens
            
        Returns:
            Output tensor after passing through all encoder layers
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            
        return self.norm(output)

class EncoderLayer(nn.Module):
    """
    Single encoder layer consisting of self-attention and feed-forward network
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.size = d_model
        self.activation = nn.ReLU()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass of encoder layer
        
        Args:
            src: Source sequence [batch_size, src_seq_len, d_model]
            src_mask: Optional mask for source sequence
            src_key_padding_mask: Optional mask for padding tokens
            
        Returns:
            Output tensor after self-attention and feed-forward network
        """
        # Self-attention block
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, 
                             key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)  # Residual connection
        src = self.norm1(src)  # Layer normalization
        
        # Feed-forward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # Residual connection
        src = self.norm2(src)  # Layer normalization
        
        return src

class Decoder(nn.Module):
    """
    Transformer Decoder consisting of N identical layers
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm if norm is not None else nn.LayerNorm(decoder_layer.size)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Pass the input through each decoder layer in turn
        
        Args:
            tgt: Target sequence [batch_size, tgt_seq_len, d_model]
            memory: Output from encoder [batch_size, src_seq_len, d_model]
            tgt_mask: Mask for target sequence (prevents attending to future tokens)
            memory_mask: Mask for source sequence
            tgt_key_padding_mask: Mask for target padding tokens
            memory_key_padding_mask: Mask for source padding tokens
            
        Returns:
            Output tensor after passing through all decoder layers
        """
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
            
        return self.norm(output)

class DecoderLayer(nn.Module):
    """
    Single decoder layer consisting of self-attention, cross-attention, and feed-forward network
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.size = d_model
        self.activation = nn.ReLU()
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of decoder layer
        
        Args:
            tgt: Target sequence [batch_size, tgt_seq_len, d_model]
            memory: Output from encoder [batch_size, src_seq_len, d_model]
            tgt_mask: Mask for target sequence (prevents attending to future tokens)
            memory_mask: Mask for source sequence
            tgt_key_padding_mask: Mask for target padding tokens
            memory_key_padding_mask: Mask for source padding tokens
            
        Returns:
            Output tensor after self-attention, cross-attention, and feed-forward network
        """
        # Self-attention block
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)  # Residual connection
        tgt = self.norm1(tgt)  # Layer normalization
        
        # Cross-attention block (attends to encoder outputs)
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                              key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)  # Residual connection
        tgt = self.norm2(tgt)  # Layer normalization
        
        # Feed-forward block
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)  # Residual connection
        tgt = self.norm3(tgt)  # Layer normalization
        
        return tgt

def create_mask(src, tgt, pad_idx):
    """
    Create masks for transformer training
    
    Args:
        src: Source tensor [batch_size, src_len]
        tgt: Target tensor [batch_size, tgt_len]
        pad_idx: Padding token index
        
    Returns:
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    """
    # src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    
    # Mask to prevent attention to future tokens in target sequence (causal mask)
    tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=src.device) * float('-inf'), diagonal=1)
    
    # Source sequence doesn't need causal mask
    src_mask = None
    
    # Padding masks
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



