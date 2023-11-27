from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import os


def get_all_sentences(ds: list, lang: str):
    for item in ds:
        yield item[lang]


def get_or_build_tokenizer(config: dict, ds: list, lang: str) -> Tokenizer:
    """
    Build or load the tokenizer from a file
    :param config: Configuration file
    :param ds: List of sentences in language
    :param lang: name of language
    :return:
        Tokenizer
    """
    tok_path = Path(str(config['tokenizer_file']).format(lang))

    if not Path.exists(tok_path):
    
        print(tok_path)

        # make directories
        path = str(tok_path).split(str(os.sep))
        subfolders = Path(f'{os.sep}'.join(path[:-1]))
        if not os.path.exists(subfolders):
            os.makedirs(subfolders)

        # build tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token= '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tok_path))

    else:
        tokenizer = Tokenizer.from_file(str(tok_path))

    return tokenizer
