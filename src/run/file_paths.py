from pathlib import Path


def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'

    return str(Path('.')/model_folder/model_filename)


def latest_weights_file_path(config):
    model_folder = 'weights'
    model_filename = f'{config["model_basename"]}*'
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
