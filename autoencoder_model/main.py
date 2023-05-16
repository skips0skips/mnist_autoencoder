import os
from pathlib import Path
from autoencoder_model.config import Config
from autoencoder_model.data_preparation import Data
from autoencoder_model.file_handler import FileHandler
from autoencoder_model.sampling import Sampling
from autoencoder_model.train_pipeline import Trainer


def start(number):

    config = Config.get_config()

    test_loader, train_loader = Data.data_loader()

    output_images_bool, val_fit_bool, n_epochs = Config.get_config_params_train()

    model_path = str(Path('autoencoder_model', 'file','model.pkl'))  

    if os.path.isfile(model_path):
        model_train = Trainer(*config, test_loader, train_loader)
        model_train.train(output_images_bool, val_fit_bool, n_epochs)
        model = Trainer.model
    else:
        model = FileHandler.load_file(model_path)

    _ ,test_dataset = Data.data_set()
    letent_vector = Sampling(test_dataset, device=Config.get_device(), model=model)

    letent_vector_path = str(Path('autoencoder_model', 'file','latent_vector.pkl')) 

    if os.path.isfile(letent_vector_path):
        latent_vector = Sampling.get_latent_vector(number)
    else:
        latent_vector = FileHandler.load_file(letent_vector_path)

    image = Sampling.get_sample(latent_vector,number,output_images_bool)

if __name__ == '__main__':
    start(1)
