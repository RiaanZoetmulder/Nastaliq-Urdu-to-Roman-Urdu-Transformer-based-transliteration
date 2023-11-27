import torch
import torch.nn as nn
from src.models.architectures import Decoder, Encoder
from src.models.embeddings import InputEmbedding, PositionalEncoding
from src.models.layers import TransformerPredictionLayer

class Seq2SeqTransformer(nn.Module):

    def __init__(
            self,
            encoder: Encoder, decoder: Decoder, source_embedding: InputEmbedding, target_embedding: InputEmbedding,
            source_position: PositionalEncoding, target_position: PositionalEncoding,
            prediction_layer: TransformerPredictionLayer ) -> None:

        super().__init__()

        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position = source_position
        self.target_position = target_position
        self.prediction_layer = prediction_layer

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, source_lang, source_mask) -> torch.Tensor:

        emb = self.source_embedding(source_lang)
        emb = self.source_position(emb)
        return self.encoder(emb, source_mask)


    def decode(self, encoder_out, source_mask, target_lang, target_mask) -> torch.Tensor:

        tgt_emb = self.target_embedding(target_lang)
        tgt_emb = self.target_position(tgt_emb)

        return self.decoder(tgt_emb, encoder_out, source_mask, target_mask)

    def predict(self, x):
        return self.prediction_layer(x)



