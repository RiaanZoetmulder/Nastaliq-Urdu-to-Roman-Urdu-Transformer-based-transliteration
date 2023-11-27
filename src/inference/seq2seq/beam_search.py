from src.models.architectures import Seq2SeqTransformer
from src.inference.masks import causal_mask, batch_causal_mask
from tokenizers import Tokenizer
import torch
from torch import Size, Tensor, BoolTensor
import torch.nn.functional as F


def unravel_index(indices: Tensor, shape: Size) -> Tensor:
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.

    Source: https://github.com/francois-rozet/torchist/blob/master/torchist/__init__.py

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]


def beam_search(model: Seq2SeqTransformer, source: torch.Tensor,
                source_mask: torch.Tensor, tokenizer_tgt: Tokenizer,
                max_len: int, device: str, beam_width: int = 10) -> torch.Tensor:
    """
    Pytorch implementation of a simple beam search for seq2seq weights.

    :param model: Sequence2Sequence model
    :param source: encoded source text
    :param source_mask: Source Mask
    :param tokenizer_tgt: Tokenizer of the target language
    :param max_len: maximum sequence length
    :param device: Device on which to run the model
    :param beam_width: Beam width
    :return:
        Torch Tensor
    """
    # get the corresponding ids to the start and stop tokens
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # only calculate the decoder output once!
    encoder_output = model.encode(source, source_mask)

    # intialize the beam
    decoder_input = torch.empty((beam_width, 1)).fill_(sos_idx).type_as(source).to(device)
    log_likelihood = torch.zeros((beam_width, 1)).to(torch.float64).to(device)

    curr_stop_tokens = 0

    while True:

        # if we reach the maximum length or all of the sentences in our beam have [STOP] token,
        # stop the inference
        if decoder_input.size(1) == max_len:
            break

        if curr_stop_tokens >= beam_width:
            break

        # build the mask
        decoder_mask = batch_causal_mask(beam_width, decoder_input.size(1)).type_as(source_mask).to(device) # (beam width, seq_len, seq_len)

        # caculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) # (beam_width, vocabulary)

        # get the probabilities of the next token
        log_prob = model.predict(out[:, -1])

        # add the log_likelihood of the current sentences to the new proposals
        log_prob = log_likelihood.repeat(1, log_prob.shape[-1]) - log_prob

        # get the indices and probabilities of the k greatest word probabilities in each INDIVIDUAL "beam"
        l_probs, indices = [], []
        for i in range(beam_width):
            l_p, ind = torch.topk(log_prob[i, :].unsqueeze(0), beam_width, largest = False) # beam_width, k && beam_width, k
            l_probs.append(l_p)
            indices.append(ind)

        # concatenate the log probabilities back into a beam width x beam width matrix
        top_kbyk_logprobs = torch.cat(l_probs, dim = 0)
        top_kbyk_indices  = torch.cat(indices, dim = 0)
        idc_shp = top_kbyk_indices.shape

        # get the coordinates of the maximum over ALL predicted words over ALL beams
        top_k_probs, top_k_indices = torch.topk(top_kbyk_logprobs.flatten(), beam_width, largest = False)
        coordinates = unravel_index(top_k_indices, idc_shp) # beam_width, 2

        # initialize temporary tensors to store beam search results in
        temp_sentences = torch.zeros_like(decoder_input).type_as(source).to(device)
        new_token = torch.zeros((beam_width, 1)).type_as(source).to(device)

        # iterate over all of the coordinates that correspond to the min log-likelihood
        for i in range(beam_width):

            # get the coordinates of the maxima
            row, col = coordinates[i, :].tolist()
            row, col = int(row), int(col)

            # extract the right values, update the log probability for each beam, the new token and the beams from
            # the previous iteration that we are keeping
            temp_sentences[i, :] = decoder_input[i, :]
            new_token[i] = top_kbyk_indices[row, col]
            log_likelihood[i] = top_kbyk_logprobs[row, col]

        # concatenate new tokens to their corresponding beams
        decoder_input = torch.cat(
            [temp_sentences, new_token], dim = 1)

        # count stop tokens
        curr_stop_tokens = (decoder_input == eos_idx).sum()

    # get the minimum log likelihood value (greatest probability sentence )
    idx = torch.argmin(log_likelihood.squeeze(1))

    return decoder_input[idx, :].squeeze(0)







        







