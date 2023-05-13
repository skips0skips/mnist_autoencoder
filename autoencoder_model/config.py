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
        model = CVAE().to(get_device())
        return model
    
    @staticmethod
    def get_optimizer():
        optimizer = torch.optim.Adam(get_model().parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
    
    def get_config():
         # Вывод изображений при обучении
        output_images_bool = False
        # Проверка на валидационной выборке
        val_fit_bool = True

        model = get_model()

        criterion = get_criterion()

        optimizer = get_optimizer()

        return model, criterion, optimizer, val_fit_bool, output_images_bool

    
    
    
   
   