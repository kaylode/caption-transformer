from .transformer import Transformer
from .captioning import Captioning

def get_transformer_model(img_size, patch_size, trg_vocab):

    transformer_config = {
        'img_size':         img_size, 
        'patch_size':       patch_size,
        'trg_vocab':        trg_vocab, 
        "d_model":          512, 
        "d_ff":             1024,
        "N_enc":            6,
        "N_dec":            6,
        "heads":            8,
        "dropout":          0.3,
        "num_channels":     3
    }

    return Transformer(**transformer_config)