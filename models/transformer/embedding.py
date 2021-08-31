import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class PatchEmbedding(nn.Module):
    """
    Image patch embedding, adapted from VIT
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2) # BCHW -> BNC
        return x

class Embeddings(nn.Module):
    """
    Word Embeddings after Tokenized
    :input:
        model_dim:    one token embedding shape
        vocab_size:   vocabulary size
    
    :output:
        embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, vocab_size, model_dim):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, model_dim)
        self.model_dim = model_dim

    def forward(self, x):
        """
        x: tokenized [batch size * text length]
        """
        return self.lut(x) * math.sqrt(self.model_dim)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding to add information to Embeddings
    :input:
        model_dim:      one token embedding shape
        dropout_rate:   dropout prob
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, model_dim, dropout_rate, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) *
                             -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add additional positional information to embedding
        x: [batch * text length * model_dim]
        """
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class FeatureEmbedding(nn.Module):
    """
    Projects image features into a space of
    dimensionality `embed_dim`.
    """

    def __init__(self, features_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(features_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)


class SpatialEncoding(nn.Module):
    """
    Encodes bounding box coordinates and relative sizes
    as vector of dimensionality `embed_dim`.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(5, embed_dim)

    def forward(self, x):
        return self.linear(x)