from .transformer import Transformer
from .captioning import Captioning

def get_transformer_model(patches_dim, trg_vocab):

    transformer_config = {
        'patches_dim':      patches_dim, 
        'trg_vocab':        trg_vocab, 
        "d_model":          512, 
        "d_ff":             1024,
        "N_enc":            6,
        "N_dec":            6,
        "heads":            8,
        "dropout":          0.3
    }

    return Transformer(**transformer_config)