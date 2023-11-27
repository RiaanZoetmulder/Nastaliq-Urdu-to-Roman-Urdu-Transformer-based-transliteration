import torch
from pathlib import Path
from .get_dataloader import get_ds
from .build_model import get_model
from torch.utils.tensorboard import SummaryWriter

from .file_paths import get_weights_file_path, latest_weights_file_path
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random


from .test import batch_run_testing
from .validation import batch_run_validation


def train_model(config: dict):
    """
    Train the nastaliq urdu to romanized urdu machine translation attention transformer
    :param config:
    """
    # ensure replicability by setting seed numbers
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # check the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # make the folder to store the model in
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # get the dataloaders and tokenizers
    train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # setup tensorboard
    writer = SummaryWriter(config['experiment_name'])
    optim = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9 )

    initial_epoch = 0
    global_step = 0

    # load model if exists
    prl = config['preload']
    model_filename = latest_weights_file_path(config) if prl == 'latest' else get_weights_file_path(config, prl) if prl else None
    if model_filename:
        print(f'Preloading model: {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optim.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        model.load_state_dict(state['model_state_dict'])

    else:
        print("No model preloaded.")

    # the loss function
    loss_func = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1)

    # iterate over epochs
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing Epoch {epoch:02d}')

        # iterate over the batches
        l = 0
        for batch in batch_iterator:

            # get model inputs, lables and masks
            encoder_input = batch['encoder_input'].to(device) # B, Seq len
            decoder_input = batch['decoder_input'].to(device) # B, Seq len
            encoder_mask = batch['encoder_mask'].to(device) # B , 1, 1, Seq len
            decoder_mask = batch['decoder_mask'].to(device) # B, 1, Seq len, Seq len
            label = batch['label'].to(device) # B, Seq Len

            # encode the encoder inputs and the decoder inputs. Make prediction
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.predict(decoder_output)

            # caculculate the loss
            loss = loss_func(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            # backprop
            loss.backward()

            # print loss in tqdm bar
            batch_iterator.set_postfix({f'loss':f'{loss.item():6.3f}'})

            # to tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # optimizer
            optim.step()
            optim.zero_grad()

            global_step +=1
            l+= 1

            if l % 20 == 0:
                torch.cuda.empty_cache()

        # Run the validation, once per epoch
        batch_run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, global_step, writer)

        # save the model at the end of each epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'global_step': global_step
        }, model_filename)

    # Run the testing
    batch_run_testing(model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device )
