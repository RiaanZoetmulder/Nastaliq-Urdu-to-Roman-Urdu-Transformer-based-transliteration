from torch.utils.data import DataLoader, random_split
from .build_tokenizer import get_or_build_tokenizer
from tokenizers import Tokenizer
from typing import Dict, Any, List

from datasets import load_dataset
from src.dataset import BilingualDataset
import os
from pathlib import Path
import json


def get_ds(config: Dict[str, Any]) -> List[Any]:
    path = os.getcwd() / Path('data/preprocessed') / Path('data.json')

    with open(str(path), 'r') as fl:
        ds_raw = json.load(fl)


    # build the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # split the dataset
    train_ds_size = int(0.8 * len(ds_raw))
    val_ds_size = int(0.02 * len(ds_raw))
    test_ds_size = len(ds_raw) - train_ds_size - val_ds_size

    train_ds_raw, val_ds_raw, test_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size, test_ds_size])

    # create the datasets
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config["lang_src"], config['lang_tgt'], config['seq_len'])

    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                                config["lang_src"], config['lang_tgt'], config['seq_len'])

    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt,
                                config["lang_src"], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids  = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids  = tokenizer_tgt.encode(item[config['lang_tgt']]).ids

        max_len_src = max(len(src_ids), max_len_src)
        max_len_tgt = max(len(tgt_ids), max_len_tgt)

    print(f'Max Length source: {max_len_src}')
    print(f'Max Length target: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size= config['batch_size'], shuffle = True, num_workers = 0)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'],  shuffle = True)
    test_dataloader = DataLoader(test_ds, batch_size=config['batch_size'],  shuffle = True)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt
