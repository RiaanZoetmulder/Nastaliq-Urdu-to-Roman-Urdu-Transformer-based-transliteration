import os
from .build_tokenizer import get_or_build_tokenizer
import json
from .get_dataloader import BilingualDataset
from torch.utils.data import DataLoader
import torch
from .build_model import get_model
from tqdm import tqdm
from src.inference import beam_search


def load_new_data(path):

    data = False
    if os.path.exists(path):
        with open(str(path), 'r') as fl:
            data = json.load(fl)

    return data


def inference(config):
    """
    Inference pipeline, uses beam search
    :param config:
    """
    # load the data
    data = load_new_data(config.IN_FOLDER / config.IN_FILE_NAME)

    # if no data found, raise error
    if not data:
        raise ValueError('Please provide input data in JSON Format')

    # load the tokenizer
    src_tokenizer = get_or_build_tokenizer(config, data, config.SRC_LANG_NAME + '_urdu')
    tgt_tokenizer = get_or_build_tokenizer(config, data, config.TGT_LANG_NAME + '_urdu')

    # check the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # if no weights, raise error
    if not os.path.exists(config['model_folder']):
        raise ValueError('No pretrained weights available ')

    # get the dataset/dataloader
    dataset = BilingualDataset(data, src_tokenizer, src_tokenizer,
                                str(config.SRC_LANG_NAME), str(config.SRC_LANG_NAME), config.SEQ_LEN)
    d_loader = DataLoader(dataset, batch_size=1,  shuffle = False)

    # set up model
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    # load the latest model
    if config.WEIGHTS_PATH:
        print(f'Preloading model: {config.WEIGHTS_PATH}')
        state = torch.load(config.WEIGHTS_PATH )
        model.load_state_dict(state['model_state_dict'])

    else:
        print("No model preloaded.")
        raise ValueError('Make sure the model is trained')

    transliteration = []
    model.eval()
    batch_iterator = tqdm(d_loader, desc = f'INFERENCE')
    with torch.no_grad():

        # run inference sentence by sentence
        for batch in batch_iterator:

            # get the inputs
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            target_text = batch['tgt_text'][0]

            # use BEAM SEARCH inference method to predict the transliteration
            model_out = beam_search(model, encoder_input, encoder_mask, src_tokenizer, config.SEQ_LEN, device)

            # decode the tokens which were calculated 
            model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            out = {
                str(config.SRC_LANG_NAME): target_text,
                str(config.TGT_LANG_NAME): model_out_text
            }

            transliteration.append(out)

    # Make the out folder and write results to file
    if not os.path.exists(config.OUT_FOLDER):
        os.makedirs(config.OUT_FOLDER)
    
    print(str(config.OUT_FOLDER / config.OUT_FILE_NAME) )
    with open(config.OUT_FOLDER / config.OUT_FILE_NAME , 'w') as out_fl:
        json.dump(transliteration , out_fl)
