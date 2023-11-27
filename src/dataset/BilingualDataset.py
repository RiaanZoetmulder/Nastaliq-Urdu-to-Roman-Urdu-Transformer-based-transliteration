import torch
from torch.utils.data import Dataset
from typing import Any
import numpy as np
from src.inference.masks import causal_mask
from typing import List, Dict
from tokenizers import Tokenizer

class BilingualDataset(Dataset):

    def __init__(self, ds: List[Dict], tokenizer_src: Tokenizer,
                 tokenizer_tgt: Tokenizer, src_lang: str, tgt_lang:str, seq_length: int) -> None:
        """
        Pytorch Dataset which tokenizes two languages, source and target.

        :param ds: List of two dicts with entries for src_lang and tgt_lang
        :param tokenizer_src: HuggingFace Tokenizer for the source language
        :param tokenizer_tgt: HuggingFace Tokenizer for the target language
        :param src_lang: source language name
        :param tgt_lang: target language name
        :param seq_length: Maximum sequence lenght of the sentences
        """
        super(BilingualDataset, self).__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_length = seq_length

        # get the start of sentence, end of sentence and padding token id
        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:

        # get the text
        src_target_pair = self.ds[index]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # encode the input sentence by using the tokenizers
        src_input_encodings = np.array(self.tokenizer_src.encode(src_text).ids)
        tgt_input_encodings = np.array(self.tokenizer_tgt.encode(tgt_text).ids)

        # calculate the amount of padding needed
        src_num_padding = self.seq_length  - len(src_input_encodings) - 2
        tgt_num_padding = self.seq_length  - len(tgt_input_encodings) - 1

        if src_num_padding < 0 or tgt_num_padding < 0:
            raise ValueError('sentence too long')

        # create the source input and target input and label tensors
        src_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src_input_encodings, dtype = torch.int64),
                self.eos_token,
                torch.tensor(src_num_padding * [self.pad_token], dtype = torch.int64)
            ]
        )

        tgt_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_input_encodings, dtype = torch.int64),
                torch.tensor(tgt_num_padding * [self.pad_token], dtype = torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(tgt_input_encodings, dtype = torch.int64),
                self.eos_token,
                torch.tensor(tgt_num_padding * [self.pad_token], dtype = torch.int64)
            ]
        )

        assert src_input.size(0) == self.seq_length
        assert tgt_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length

        return {
            'encoder_input': src_input,
            'decoder_input': tgt_input,
            'encoder_mask': (src_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (tgt_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(tgt_input.size(0)),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
