import torch
from src.inference import batch_greedy_decode, greedy_decode
import torchmetrics
from tqdm import tqdm
from src.models.architectures import Seq2SeqTransformer
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter


def batch_run_validation(model: Seq2SeqTransformer, validation_ds: DataLoader, tokenizer_src:Tokenizer,
                         tokenizer_tgt: Tokenizer, max_len:int, device: str, global_step: int,
                         writer: SummaryWriter) -> None:
    """
    Validate Seq2SeqTransformer on batches of data

    :param model: Seq2SeqTransformer model to validate
    :param validation_ds: Pytorch dataset with validation data
    :param tokenizer_src: source tokenizer
    :param tokenizer_tgt: target tokenizer
    :param max_len: maximum sequence lenght
    :param device: Device on which to perform calculations
    :param global_step: Global step during training
    :param writer: SummaryWriter object
    """

    # put model into eval mode
    model.eval()

    # End of sentence token id
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # save expected and predicted
    expected = []
    predicted = []

    # make tqdm progress bar
    batch_iterator = tqdm(validation_ds, desc = 'VALIDATING')

    # print function with the current tqdm context
    print_msg = lambda msg: batch_iterator.write(msg)

    # validate whilst not using gradients
    with torch.no_grad():
        k = 0
        for batch in batch_iterator:

            k+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            target_text = batch['tgt_text']

            # make prediction
            model_out = batch_greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            # Remove padding tokens, if any
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

            # print first few examples
            if k < 2:
                j = 0
                for e, p in zip(expected, predicted):

                    if j > 4: break

                    print_msg(f'ACTUAL: {e}')
                    print_msg(f'PREDIC: {p}')
                    print_msg('-' * 80 + '\n')
                    j+=1

    # calculate the examples
    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation Char Error Rate', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation Word Error Rate', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU Score', bleu, global_step)
        writer.flush()


