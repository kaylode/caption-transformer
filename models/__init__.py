from .transformer import Transformer, TransformerBottomUp
from .captioning import Captioning

def get_transformer_model(trg_vocab):

    transformer_config = {
        'trg_vocab':        trg_vocab, 
        "d_model":          512, 
        "d_ff":             3072,
        "N_enc":            6,
        "N_dec":            6,
        "heads":            8,
        "dropout":          0.3,
        "num_channels":     3,
        'pretrained_encoder': True
    }

    return Transformer(**transformer_config)

def get_transformer_bottomup_model(bottom_up_dim, trg_vocab):

    transformer_config = {
        'feat_dim':       bottom_up_dim,
        'trg_vocab':        trg_vocab, 
        "d_model":          768, 
        "d_ff":             3072,
        "N_enc":            6,
        "N_dec":            6,
        "heads":            8,
        "dropout":          0.3,
    }

    return TransformerBottomUp(**transformer_config)