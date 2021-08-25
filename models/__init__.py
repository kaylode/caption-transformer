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

def get_lstm_model(src_vocab, trg_vocab):
    lstm_config = {
            'src_vocab':        src_vocab, 
            'trg_vocab':        trg_vocab, 
            "embed_dim":        512, 
            "hidden_dim":       1024,
            "num_layers":       1,
            'bidirectional':    False,
            'dropout' :         None
        }

    return Seq2Seq(**lstm_config)