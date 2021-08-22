from .transformer import Transformer
from .seq2seq import Seq2Seq
from .captioning import Captioning

def get_transformer_model(patches_dim, trg_vocab):

    transformer_config = {
        'patches_dim':      patches_dim, 
        'trg_vocab':        trg_vocab, 
        "d_model":          768, 
        "d_ff":             2048,
        "N_enc":            12,
        "N_dec":            4,
        "heads":            8,
        "dropout":          0.2
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