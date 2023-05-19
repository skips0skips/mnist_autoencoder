import os
from pathlib import Path
from autoencoder_model.config import Config
from autoencoder_model.data_preparation import Data
from autoencoder_model.file_handler import FileHandler
from autoencoder_model.model import CVAE
from autoencoder_model.sampling import Sampling
from autoencoder_model.train_pipeline import Trainer

#python -m autoencoder_model.main
class Main():
    def __init__(self):
        self.model, self.criterion, self.optimizer, self.device = Config.get_config_model()
        data = Data()
        self.test_loader, self.train_loader = data.data_loader()
        self.output_images_bool, self.val_fit_bool, self.n_epochs = Config.get_config_params_train()
        self.train_fit_bool = False # При значении True модель будет обучаться

    def get_model(self,):
        '''Метод получает веса модели. Если веса модели сохранены то обучения не будет, кроме тех случаев где переменная train_fit_bool
        равна True'''
        # model, criterion, optimizer, device = Config.get_config_model()
        # data = Data()
        # test_loader, train_loader = data.data_loader()
        # output_images_bool, val_fit_bool, n_epochs = Config.get_config_params_train()

        model_path = str(Path('autoencoder_model', 'file','model'))  

        if not os.path.isfile(model_path) or self.train_fit_bool == True:
            model_train = Trainer(self.model, self.criterion, self.optimizer, self.device, self.test_loader, self.train_loader)
            model_train.train(self.output_images_bool, self.val_fit_bool, self.n_epochs)
            self.model = Trainer.model
            print('Конец обучения')
        else:
            CVAE()
            self.model = FileHandler.load_file2('model')
            print('Модель загружена')
        

    def create_image(self,number):
        '''Метод создает изображения цифры number которую указал пользователь. Если латентного пространства нет оно будет составлено.'''    
        letent_vector_path = str(Path('autoencoder_model', 'file','latent_vector.pkl')) 
        _ ,test_dataset = Data.data_set()
        if os.path.isfile(letent_vector_path):
            letent_vector = Sampling(test_dataset, device=Config.get_device(), model=self.model)
            latent_vector = Sampling.get_latent_vector(number)
        else:
            latent_vector = FileHandler.load_file2('latent_vector')

        sample = Sampling(test_dataset, self.device, self.model)
        image = sample.get_sample(latent_vector,number,self.output_images_bool)

    def get_image(self, number):
        '''Метод выдает полученные изображения'''
        image = Sampling.load_image(number)
        return image
    
    def start(self, number):
        '''Метод запускает получение изображений
        number - это цифра от 0 до 9 которую хочет сгенерировать пользователь'''
        self.get_model()
        self.create_image(number)
    
    def get_config(self):
        '''Метод выводит пользователю текущие параметры оптимайзера, устройства запуска'''
        return  self.optimizer, self.device, self.output_images_bool, self.val_fit_bool, self.n_epochs
    
    def set_config(self, train_fit_bool=None, val_fit_bool=None, n_epochs=None, output_images_bool=None):
        '''Метод изменяет значения: 
        train_fit_bool Обучать модель: True - обучать, False - не обучать
        val_fit_bool Проверка на валидационных данных при обучении: True - проверять, False - не проверять
        n_epochs Количество эпох при обучении: Int
        output_images_bool Выводить изображение пользователю: True - выводить, False - не выводить
        '''
        self.train_fit_bool = self.train_fit_bool if train_fit_bool is None else train_fit_bool
        self.val_fit_bool = self.val_fit_bool if val_fit_bool is None else val_fit_bool
        self.n_epochs = self.n_epochs if n_epochs is None else n_epochs
        self.output_images_bool = self.output_images_bool if output_images_bool is None else output_images_bool
    def set_initial_phrase():
        '''Метод выводит начальную фразу'''
        return str('Это модель')
        
if __name__ == '__main__':
    main = Main()
    main.start(1)
