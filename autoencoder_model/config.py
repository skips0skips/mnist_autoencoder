import torch
from autoencoder_model.func_loss import loss_vae

from autoencoder_model.model import CVAE


class Config:

    @staticmethod
    def get_criterion():
        criterion = loss_vae
        return criterion
 
    @staticmethod
    def get_device():
        device = torch.device('cpu')#'cuda:0' if torch.cuda.is_available() else 'cpu'
        return device
    
    @staticmethod
    def get_model():
        model = CVAE().to(Config.get_device())
        return model
    
    @staticmethod
    def get_optimizer():
        optimizer = torch.optim.Adam(Config.get_model().parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
    
    def get_config_model():

        model = Config.get_model()

        criterion = Config.get_criterion()

        optimizer = Config.get_optimizer()

        device = Config.get_device()

        return model, criterion, optimizer, device
    
    def get_config_params_train():
        n_epochs = 50
        # Вывод изображений при обучении
        output_images_bool = False
        # Проверка на валидационной выборке
        val_fit_bool = True
        
        return output_images_bool, val_fit_bool, n_epochs

    
    
    
   
   