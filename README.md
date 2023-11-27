# Nastaliq Urdu to Informal Romanized Urdu Transliteration Using a Language Transformer
## Description 
The project is about romanizing Urdu written in [Nastaliq](https://en.wikipedia.org/wiki/Nastaliq) scipt. Though a formal system for romanizing nastaliq urdu exists, it uses characters other than the 26 letters of the latin alphabet. Moreover, on the internet nastaliq is often romanized using only the 26 letters of the roman alphabet without the use of formal rules. I wanted transliterate it into the core 26 latin characters to help with langauge learning. In this project I built a transformer model to accomplish this task. This github repository contains the code to preprocess the data, train a simple language [transformer model](https://arxiv.org/pdf/1706.03762.pdf), and containerize the model by using Docker. 

## Installation
Install anaconda and make an anaconda environment. Make sure pip is installed and that you are using Python 3.11. Activate the anaconda environment and run:

```sh
python -m pip install --user -r requirements.txt
```
## Set up the file structure

Make the following folders:.
├── ...
├── data                            
│   ├── raw                         # Raw data from the Dakshina dataset and roman urdu parl go here
│   ├── preprocessed                # After preprocessing, the data will live here
│   └── tokenizers                  # Tokenizer files live here after preprocessing
└── input
│   ├── in_text                     # Folder that will contain the nastaliq_urdu.json file, i.e. the new data that we want to get romanized. This folder will be mounted to the docker container
└── output
│   ├── transliteration             # Folder that contains the transliterated output from the transformer models after inference. Will be a .json file with this structure: [{'nastaliq': ..., 'roman': ...}, ...]
└── weights                         # checkpoints folder for saved models
│   ├── tmodel_##.pt               
└── runs                            # Folder that contains the tensorboard files.
│   ├── file.ckpt               
└── src                             # Source code
│   ├── tmodel_##.pt               

### Get & Clean the Data

We use two different data sources in this project:
1. The Dakshina Dataset([paper](https://arxiv.org/abs/2007.01176), [data](https://github.com/google-research-datasets/dakshina)) 
2. Roman Urdu Parl ([paper](https://dl.acm.org/doi/fullHtml/10.1145/3464424), [data](https://drive.google.com/drive/folders/1yXzE8ejq7EZxIsumIsEbkHdv9eAKUGs9))

Make a folder called "data/raw/" and put the urdu.txt and roman-urdu.txt files from the Roman Urdu Parl dataset in it. Next, in the Dakshina dataset go to the "ur/romanized" folder and copy the files ur.romanized.rejoined.dev.native.txt, ur.romanized.rejoined.dev.roman.txt, ur.romanized.rejoined.test.native.txt, and ur.romanized.rejoined.test.roman.txt to  "data/raw/".

To clean the data run the following command:

```sh
python main.py --mode=preprocess
```

A preprocessed folder should now appear in the data folder with the cleaned dataset (about 2.7 GB)

### Train the model

Make sure you have a GPU with about 12 GB of VRAM and at least 12 GB RAM. My results were obtained by using a computer with an NVIDIA Geforce RTX 3060, 16 GB of RAM, and an Intel i5-6600K CPU running Linux 23.04 Lunar. Make sure you have your environment set up and have preprocessed the data. Run the following command:

```sh
python main.py --mode=train
```
It should start training and it takes about 3 hours per epoch. All the hyperparameters can be found in the main.py file. 
The tensorboard logs can be found in the "runs" folder, the models can be found in the "weights" folder. 

### Containerize
Make sure that you have trained the model. Make the following folders:
- input/in_text
- output/transliteration
- data/tokenizers

Make sure Docker is installed. Within the docker container, input data will be mounted to "opt/algorithm/input/in_text/nastaliq_urdu.json", the output folder will be mounted to "opt/algorithm/output/transliteration". Remove any of the weights that you don't want to use to save disc space. Run the following commands:

```sh
sudo docker build . -t urdu_transliteration
```

To run the model make sure you save your data in a .json file and put it into the "opt/algorithm/input/in_text/" folder. Make sure to save it as a json file as a list of dicts: [{'nastaliq': ...}, ...]

```sh
sudo docker run --gpus all -v $PWD/input/in_text/:/opt/algorithm/input/in_text/ -v $PWD/output/transliteration/:/opt/algorithm/input/output/transliteration/ urdu_transliteration:latest /bin/bash
```

# References
- Alam, M., & Hussain, S. U. (2022). Roman-Urdu-Parl: Roman-Urdu and Urdu Parallel Corpus for Urdu Language Understanding. Transactions on Asian and Low-Resource Language Information Processing, 21(1), 1-20.
- Roark, B., Wolf-Sonkin, L., Kirov, C., Mielke, S. J., Johny, C., Demirsahin, I., & Hall, K. (2020). Processing South Asian languages written in the Latin script: the Dakshina dataset. arXiv preprint arXiv:2007.01176.

Special thanks to Umar Jamil who created a wonderful video about programming a transformer model: https://www.youtube.com/watch?v=ISNdQcPhsts

