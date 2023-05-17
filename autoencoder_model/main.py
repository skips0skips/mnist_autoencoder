import os
from pathlib import Path
from autoencoder_model.config import Config
from autoencoder_model.data_preparation import Data
from autoencoder_model.file_handler import FileHandler
from autoencoder_model.model import CVAE
from autoencoder_model.sampling import Sampling
from autoencoder_model.train_pipeline import Trainer

#python -m autoencoder_model.main

def start(number):

    model, criterion, optimizer, device = Config.get_config_model()

    data = Data()
    test_loader, train_loader = data.data_loader()

    output_images_bool, val_fit_bool, n_epochs = Config.get_config_params_train()

    model_path = str(Path('autoencoder_model', 'file','model.pkl'))  

    if os.path.isfile(model_path):
        model_train = Trainer(model, criterion, optimizer, device, test_loader, train_loader)
        model_train.train(output_images_bool, val_fit_bool, n_epochs)
        model = Trainer.model
    else:
        CVAE()
        model = FileHandler.load_file2('model')

    _ ,test_dataset = Data.data_set()

    letent_vector_path = str(Path('autoencoder_model', 'file','latent_vector.pkl')) 

    if os.path.isfile(letent_vector_path):
        letent_vector = Sampling(test_dataset, device=Config.get_device(), model=model)
        latent_vector = Sampling.get_latent_vector(number)
    else:
        latent_vector = FileHandler.load_file2('latent_vector')

    sample = Sampling(test_dataset, device, model)
    image = sample.get_sample(latent_vector,number,output_images_bool)
    print('Конец')

if __name__ == '__main__':
    start(5)
