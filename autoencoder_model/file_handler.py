from pathlib import Path
import pickle
import os
import joblib


class FileHandler:
    folder_path = str(Path('autoencoder_model', 'file'))

    @staticmethod
    def save_file(data, filename):
        file_path = os.path.join(FileHandler.folder_path, filename)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def load_file(filename):
        file_path = os.path.join(FileHandler.folder_path, filename)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data