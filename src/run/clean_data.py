from pathlib import Path
import os
import json


def clean_data(path: Path = os.getcwd() / Path('data')):
    """
    Convert the data to .JSON format
    :param path:
    """

    # read the first dataset
    subdir = Path('raw')
    rom_urdu_path = path / subdir / Path('roman-urdu.txt')
    nas_urdu_path = path / subdir / Path('nastaliq_urdu.txt')

    # read the second dataset
    base_path = path / subdir
    urdu_nast_dev_path = base_path / Path('ur.romanized.rejoined.dev.native.txt')
    urdu_rom_dev_path = base_path / Path('ur.romanized.rejoined.dev.roman.txt')
    urdu_nast_test_path = base_path / Path('ur.romanized.rejoined.test.native.txt')
    urdu_rom_test_path = base_path / Path('ur.romanized.rejoined.test.roman.txt')

    out_data_path = path / Path('preprocessed') / Path('data.json')

    # read the first dataset
    file_1 = open(rom_urdu_path, 'r')
    rom_urdu_lines = file_1.readlines()

    file_2 = open(nas_urdu_path, 'r')
    nas_urdu_lines = file_2.readlines()

    # load second dakshina dataset
    with open(urdu_nast_dev_path, 'r') as f:
        urdu_nast_dev = f.readlines()

    with open(urdu_rom_dev_path, 'r') as f:
        urdu_rom_dev = f.readlines()

    with open(urdu_nast_test_path, 'r') as f:
        urdu_nast_test = f.readlines()

    with open(urdu_rom_test_path, 'r') as f:
        urdu_rom_test = f.readlines()

    # add the lines of romanized urdu
    rom_urdu_lines += urdu_rom_dev
    rom_urdu_lines += urdu_rom_test

    # add the lines of nastaliq urdu
    nas_urdu_lines += urdu_nast_dev
    nas_urdu_lines += urdu_nast_test

    out_data = []
    for ru, nu in zip(rom_urdu_lines,nas_urdu_lines):

        if len(nu.split(' ')) > 100 or len(ru.split(' ')) > 100 :
            continue

        out_dict = {
            'roman_urdu': ru,
            'nastaliq_urdu': nu
        }
        out_data.append(out_dict)

    with open(str(out_data_path), 'w') as out:
        json.dump(out_data, out)
