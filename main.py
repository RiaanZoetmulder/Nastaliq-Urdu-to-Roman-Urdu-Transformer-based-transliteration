import os
import sys
from pathlib import Path

if os.environ.get('DEPLOYED', 0):
    print(os.getcwd())
    
    print(os.listdir())
    print('-'*30)
    print(list(os.walk(str(os.getcwd() / Path('src')))))
    print('-'* 80)
    print(list(os.walk(str(os.getcwd() / Path('input')))))
    print('-'* 80)
    print(list(os.walk(str(os.getcwd() / Path('data')))))
    pkg_path = Path(
        Path(os.environ.get('WORKDIR', f'{os.getcwd()}'))
    )
    sys.path.append(
        str(pkg_path)
    )
    
from src.run.file_paths import latest_weights_file_path
import argparse

# get the relevant functions
from src.run.train import train_model
from src.run.test import test
from src.run.run_inference import inference
from src.run.clean_data import clean_data

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_config():
    return {
        'batch_size': 128,
        'num_epochs':1,
        'lr': 10**-3,
        'seq_len': 120,
        'd_model': 32,
        'lang_src':'nastaliq_urdu',
        'lang_tgt':'roman_urdu',
        'model_folder':'weights',
        'model_basename': 'tmodel_',
        'preload': 'latest',
        'tokenizer_file': 'data/tokenizers/tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel',
    }


def get_inference_config():

    parameters = AttrDict()

    parameters.CWD = CWD = Path(os.environ.get('WORKDIR', f'{os.getcwd()}'))
    parameters.IN_FOLDER = IN_FOLDER = CWD / Path('input/in_text')
    parameters.OUT_FOLDER = OUT_FOLDER = CWD / Path('output/transliteration')

    # name of the source language
    parameters.SRC_LANG_NAME = SRC_LANG_NAME = 'nastaliq'
    parameters.TGT_LANG_NAME = TGT_LANG_NAME = 'roman'

    # in file names
    parameters.IN_FILE_NAME = IN_FILE_NAME = Path(f'{SRC_LANG_NAME}_urdu.json')
    parameters.OUT_FILE_NAME = OUT_FILE_NAME = Path(f'{TGT_LANG_NAME}_urdu.json')

    # in file path
    parameters.IN_FILE_PATH = IN_FOLDER / IN_FILE_NAME
    parameters.OUT_FILE_PATH = OUT_FOLDER / OUT_FILE_NAME

    # update the configuration
    parameters.update(
        AttrDict(get_config())
    )
    
    parameters['tokenizer_file'] = CWD /Path('data/tokenizers/tokenizer_{0}.json')

    # sequence length and the weights path
    parameters.SEQ_LEN = 120
    parameters.WEIGHTS_PATH = latest_weights_file_path(parameters)

    return parameters


def main():

    parser = argparse.ArgumentParser('Train/Test/Inference for Nastaliq to Roman script')
    parser.add_argument('--mode', type = str, default = None, help = 'train/test/inference/preprocess')

    args = parser.parse_args()

    if args.mode == 'train':
        config = get_config()
        train_model(config)

    elif args.mode == 'test':
        config = get_config()
        test(config)

    elif args.mode == 'inference':
        config = get_inference_config()
        inference(config)

    elif args.mode == 'preprocess':
        clean_data()

    else:
        print('No valid argument passed')


if __name__ == '__main__':

    deployed = os.environ.get('DEPLOYED', 0)
    if deployed:
        config = get_inference_config()
        print('-'* 80)
        for key, value in config.items():
            print(f'Key: {key} \t -> \t Value: {str(value)}')
            
        inference(config)

    else:
        main()


