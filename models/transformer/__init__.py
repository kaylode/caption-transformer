import copy
import torch.nn as nn
from .embedding import Embeddings, PositionalEncoding, PatchEmbedding
from .layers import EncoderLayer, DecoderLayer
from .norm import LayerNorm
from .utils import draw_attention_map
from .search import sampling_search, beam_search

def get_clones(module, N):
    """
    "Produce N identical layers."
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_pretrained_encoder(model_name='base224'):
    import timm
    if model_name == 'tiny224':
        model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=True)
    elif model_name == 'small224':
        model = timm.create_model('deit_small_distilled_patch16_224', pretrained=True)
    elif model_name == 'base224':
        model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
    elif model_name == 'base384':
        model = timm.create_model('deit_base_distilled_patch16_384', pretrained=True)
    return model

class Encoder(nn.Module):
    """
    Core encoder is a stack of N EncoderLayers
    :input:
        patches_dim:    size of patches
        d_model:        embeddings dim
        d_ff:           feed-forward dim
        N:              number of layers
        heads:          number of attetion heads
        dropout:        dropout rate
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, patch_size, d_model, d_ff, N, heads, dropout, num_channels=3, pretrained=True):
        super().__init__()
        self.N = N

        self.pretrained = pretrained

        if not pretrained:
            self.embed = PatchEmbedding(patch_size=patch_size, in_chans=num_channels, embed_dim=d_model)
            self.pe = PositionalEncoding(d_model, dropout_rate=dropout)
            self.layers = get_clones(EncoderLayer(d_model, d_ff, heads, dropout), N)
            self.norm = LayerNorm(d_model)
        else:
            self.vit = get_pretrained_encoder()
            self.embed_dim = self.vit.embed_dim 
        
    def forward(self, src, mask):
        if not self.pretrained:
            x = self.embed(src)
            x = self.pe(x)
            for i in range(self.N):
                x = self.layers[i](x, mask)
            x = self.norm(x)
        else:
            x = self.vit.forward_features(src)
            if len(x) == 2:
                x = x[0]
        return x

class Decoder(nn.Module):
    """
    Decoder with N-stacked DecoderLayers
    :input:
        vocab_size:     size of target vocab
        d_model:        embeddings dim
        d_ff:           feed-forward dim
        N:              number of layers
        heads:          number of attetion heads
        dropout:        dropout rate
    :output:
        decoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, vocab_size, d_model, d_ff, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout_rate=dropout)
        self.layers = get_clones(DecoderLayer(d_model, d_ff, heads, dropout), N)
        self.norm = LayerNorm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Transformer model
    :input:
        patch_size:    size of patch
        trg_vocab:     size of target vocab
        d_model:       embeddings dim
        d_ff:          feed-forward dim
        N:             number of layers
        heads:         number of attetion heads
        dropout:       dropout rate
    :output:
        next words probability shape [batch * input length * vocab_dim]
    """
    def __init__(self, patch_size, trg_vocab, d_model=768, d_ff=3072, N_enc=12, N_dec=4, heads=12, dropout=0.2, num_channels=3, pretrained_encoder=True):
        super().__init__()
        self.name = "Transformer"
        self.encoder = Encoder(patch_size, d_model, d_ff, N_enc, heads, dropout, num_channels=num_channels, pretrained=pretrained_encoder)
        if pretrained_encoder:
            # Override decoder hidden dim if use pretrained encoder
            d_model = self.encoder.embed_dim
        self.decoder = Decoder(trg_vocab, d_model, d_ff, N_dec, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
        self.init_params()

    def forward(self, src, trg, src_mask, trg_mask, *args, **kwargs):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def predict(
        self, src_inputs, src_masks, 
        tokenizer, max_len=None, 
        top_k = 5, top_p=0.9, temperature = 0.9,
        *args, **kwargs):

        """
        Inference step
        """

        if max_len is None:
            max_len = src_inputs.shape[-1]

        # sampling_search, beam_search
        outputs = sampling_search(
            self, 
            src=src_inputs, 
            src_mask=src_masks, 
            max_len=max_len, 
            top_k = top_k, top_p=top_p, 
            temperature = temperature,
            tokenizer=tokenizer)

        # outputs = beam_search(
        #     self, 
        #     src=src_inputs, 
        #     src_mask=src_masks,
        #     tokenizer=tokenizer, 
        #     max_len=max_len, k=3, alpha=0.7)

        return outputs