from pathlib import Path
import pickle


def save(name):
    path = str(Path('file', name))
    with open(path, 'wb') as file:
        pickle.dump(name, file)