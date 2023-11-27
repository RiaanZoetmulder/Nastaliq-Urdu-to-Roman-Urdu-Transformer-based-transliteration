from src.model_builders import build_seq2seq_transformer

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_seq2seq_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config['seq_len'],
        config['seq_len'],
        config['d_model'],
        N = 1,
        heads = 4,
        dim_ff = 1024
    )
    return model
