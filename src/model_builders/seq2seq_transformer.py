import torch
import torch.nn as nn

from src.models.embeddings import InputEmbedding, PositionalEncoding
from src.models.attention import MultiHeadAttention
from src.models.layers import SimpleFeedForward, TransformerPredictionLayer
from src.models.blocks import EncoderBlock, DecoderBlock
from src.models.architectures import Decoder, Encoder, Seq2SeqTransformer


def build_seq2seq_transformer(source_vocab_size: int, target_vocab_size:int,
                              source_seq_length:int, target_seq_length: int, hidden_dim:int = 512,
                              N:int = 6, heads:int = 8, dropout:int = 0.1, dim_ff: int =  2048):


    # word embeddings for the source and target vocabulary
    source_embedding = InputEmbedding(hidden_dim, source_vocab_size)
    target_embedding = InputEmbedding(hidden_dim, target_vocab_size)

    # positional encodings
    source_positional_encoding = PositionalEncoding(hidden_dim, source_seq_length, dropout=dropout)
    target_positional_encoding = PositionalEncoding(hidden_dim, target_seq_length, dropout=dropout)

    # encoder blocks
    encoder_blocks = []
    for _ in range(N):

        # create the self attention
        encoder_att_block = MultiHeadAttention(hidden_dim, heads, dropout = dropout)

        # feedforward layer at the end of the encoder
        ff_block = SimpleFeedForward(hidden_dim, dim_ff, dropout = dropout)

        # create and append the encoder block to the list
        encoder_block = EncoderBlock(hidden_dim, encoder_att_block, ff_block, dropout = dropout)

        encoder_blocks.append(encoder_block)

    encoder_module_list = nn.ModuleList(encoder_blocks)

    # create the decoder block
    decoder_blocks = []
    for _ in range(N):

        # create the self attention
        decoder_self_att_block = MultiHeadAttention(hidden_dim, heads, dropout = dropout)

        # create the cross attention
        decoder_cross_att_block = MultiHeadAttention(hidden_dim, heads, dropout = dropout)

        ff_block = SimpleFeedForward(hidden_dim, dim_ff, dropout = dropout)

        decoder_block = DecoderBlock(hidden_dim, decoder_self_att_block, decoder_cross_att_block, ff_block, dropout = dropout)

        decoder_blocks.append(decoder_block)

    decoder_module_list = nn.ModuleList(decoder_blocks)

    # create encoder
    encoder = Encoder(hidden_dim, encoder_module_list)

    # create decoder
    decoder = Decoder(hidden_dim, decoder_module_list)

    # prediction layer
    pred_layer = TransformerPredictionLayer(hidden_dim, target_vocab_size)

    # create the sequence to sequence transformer
    seq2seq_transformer = Seq2SeqTransformer(encoder, decoder, source_embedding,
                                             target_embedding, source_positional_encoding,
                                             target_positional_encoding, pred_layer)

    # initialization using xavier uniform
    for param in seq2seq_transformer.parameters():
        if param.dim()>1:
            nn.init.xavier_uniform(param)

    return seq2seq_transformer
    
