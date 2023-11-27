import torch
import torchmetrics
from torchmetrics.text import CharErrorRate, WordErrorRate
from src.inference.seq2seq import batch_greedy_decode
from tqdm import tqdm
import os

from .file_paths import get_weights_file_path, latest_weights_file_path
from .build_model import get_model
import random
import numpy as np
from .get_dataloader import get_ds
from src.models.architectures import Seq2SeqTransformer
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from typing import Dict

from nltk.translate.bleu_score import corpus_bleu


def batch_run_testing(model: Seq2SeqTransformer, test_ds: DataLoader, tokenizer_src: Tokenizer,
                      tokenizer_tgt: Tokenizer, max_len: int, device: str):
    """
    Run testing of the Seq2SeqTransformer on batches
    :param model: Seq2SeqTransformer
    :param test_ds: Testing dataset
    :param tokenizer_src: Source huggingface tokenizer
    :param tokenizer_tgt: Target huggingface tokenizer
    :param max_len: Maximum length
    :param device: Device on which we are running the computation
    """
    # evaulation mode
    model.eval()

    # get end of sentence tokens's id
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # lists to store the results
    expected = []
    predicted = []

    # error metrics
    ch_err = CharErrorRate()
    w_err = WordErrorRate()

    # tqdm progress bar and printing function
    batch_iterator = tqdm(test_ds, desc = f'TESTING ')
    print_msg = lambda msg: batch_iterator.write(msg)

    with torch.no_grad():

        # iterate over batches and param k to keep track of
        k = 0
        for batch in batch_iterator:

            # get the inputs, masks and target text
            k+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            target_text = batch['tgt_text']

            # use greedy decoding method during testing
            model_out = batch_greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            # remove any extraneous padding and tokens
            for i in range(int(model_out.shape[0])):

                vals = model_out[i, :].detach().cpu().numpy()
                index = -1

                try:
                    index = list(vals).index(eos_idx)
                except ValueError:
                    pass

                if index > -1:
                    vals = vals[:index]

                predicted.append(tokenizer_tgt.decode(vals))

            expected.extend(target_text)

            # print first 2 target and predicted sentences
            if k < 2:
                j = 0
                for e, p in zip(expected, predicted):

                    if j > 4: break

                    print_msg(f'ACTUAL: {e}')
                    print_msg(f'PREDIC: {p}')
                    print_msg('-' * 80 + '\n')
                    j+=1

    # calculate the error rates
    cer = ch_err(predicted, expected)
    print_msg(f'TEST Character Error Rate: {cer}')
    wer = w_err(predicted, expected)
    print_msg(f'TEST Word Error Rate: {wer}')

    # converting predicted and target sentences to lists of tokens
    predicted = [sentence.split(' ') for sentence in predicted]
    expected= [sentence.split(' ') for sentence in expected]

    bls = corpus_bleu(predicted, expected)

    print_msg(f'TEST CORPUS BLEU: -> {bls}')





def test(config: Dict):
    """
    Load model and run testing

    :param config:
    """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # check the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(config['model_folder']):
        raise ValueError

    # get the dataloaders and tokenizers
    _, _, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # load the model
    prl = config['preload']
    model_filename = latest_weights_file_path(config) if prl == 'latest' else get_weights_file_path(config, prl) if prl else None
    if model_filename:
        print(f'Preloading model: {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])

    else:
        print("No model preloaded.")
        raise ValueError('Make sure the model is trained')

    # run testing in batches
    batch_run_testing(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
