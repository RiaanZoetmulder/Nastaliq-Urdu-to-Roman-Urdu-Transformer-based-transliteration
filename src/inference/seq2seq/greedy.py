from src.models.architectures import Seq2SeqTransformer
from src.inference import causal_mask, batch_causal_mask
from tokenizers import Tokenizer
import torch


def greedy_decode(model: Seq2SeqTransformer, source: torch.Tensor,
                  source_mask: torch.Tensor, tokenizer_tgt: Tokenizer,
                  max_len: int, device: str) -> torch.Tensor:
    """
    Greedy decoding method for seq2seq inference

    Used for a singular sentence

    :param model: Sequence2Sequence model
    :param source: encoded source text
    :param source_mask:  Source Mask
    :param tokenizer_tgt: Tokenizer of the target language
    :param max_len: maximum sequence length
    :param device: Device on which to run the model
    :return:
        torch.Tensor
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build the mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # caculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next_token
        prob = model.predict(out[:, -1])

        _, next_word = torch.max(prob, dim = 1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim = 1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def batch_greedy_decode(model: Seq2SeqTransformer, source: torch.Tensor, source_mask: torch.Tensor,
                        tokenizer_tgt: Tokenizer, max_len: int, device: str) -> torch.Tensor:
    """

    Greedy decoding method for seq2seq inference

    Used for a batch of sentences

    :param model: Sequence2Sequence model
    :param source: encoded source text
    :param source_mask: Source Mask
    :param tokenizer_tgt: Tokenizer of the target language
    :param max_len: maximum sequence length
    :param device: Device on which to run the model
    :return:
        torch.Tensor
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(encoder_output.shape[0], 1).fill_(sos_idx).type_as(source).to(device)
    curr_stop_tokens = 0
    while True:

        if decoder_input.size(1) == max_len:
            break

        if curr_stop_tokens == max_len:
            break

        decoder_mask = batch_causal_mask(decoder_input.size(0), decoder_input.size(1)).type_as(source_mask).to(device)

        # caculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next_token
        prob = model.predict(out[:, -1])

        _, next_word = torch.max(prob, dim = 1)

        decoder_input = torch.cat(
            [decoder_input, torch.unsqueeze(torch.Tensor(next_word.tolist()), 1).type_as(source).to(device)], dim = 1)

        curr_stop_tokens += (next_word == eos_idx).sum()

    return decoder_input
