from io import BytesIO
from pathlib import Path
import pickle
import os
import joblib
import torch

from autoencoder_model.model import CVAE


class FileHandler:
    folder_path = str(Path('autoencoder_model', 'file'))

    @staticmethod
    def save_file(data, filename):
        file_path = os.path.join(FileHandler.folder_path, filename)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def load_file(filename, map_location=torch.device('cpu')):
        file_path = os.path.join(FileHandler.folder_path, filename)
        with open(file_path, 'rb') as file:
            #data = pickle.load(file,map_location=map_location)
            data = torch.load(file, map_location=map_location)
        return data
    def load_file2(filename):
        loaded_object = None
        file_path = os.path.join(FileHandler.folder_path, filename)
        with open(file_path, 'rb') as file:
            serialized_data = file.read()
            loaded_object = mapped_loads(serialized_data, map_location='cpu')
        return loaded_object


def fix(map_loc):
    # Closure rather than a lambda to preserve map_loc 
    return lambda b: torch.load(BytesIO(b), map_location=map_loc)

class MappedUnpickler(pickle.Unpickler):
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219

    def __init__(self, *args, map_location='cpu', **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return fix(self._map_location)
        else: 
            return super().find_class(module, name)

def mapped_loads(s, map_location='cpu'):
    bs = BytesIO(s)
    unpickler = MappedUnpickler(bs, map_location=map_location)
    return unpickler.load()