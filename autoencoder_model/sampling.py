import numpy as np
from torch.utils.data import Subset
import torch

from autoencoder_model.data_preparation import Data
from autoencoder_model.file_handler import FileHandler
from autoencoder_model.plt_show import Plt_show

class Sampling:
  def __init__(self, test_dataset, device, model):
      self.test_dataset = test_dataset
      self.device = device
      self.model = model

  def get_latent_vector(self,number):
      '''Возьмем к примеру изображение единицы и получим латентный вектор'''
      # Индекс цифры 1 в датасете
      digit_1_indices = torch.where(self.test_dataset.targets == number)[0]

      # Создание подмножества из датасета train_dataset с индексами digit_1_indices
      digit_1_subset = Subset(self.test_dataset, digit_1_indices)

      # Создание DataLoader для подмножества digit_1_subset
      digit_1_loader = torch.utils.data.DataLoader(dataset=digit_1_subset, batch_size=Data.batch_size, shuffle=True)

      '''Получим латентный вектор цифры 1'''
      self.model.eval()
      with torch.no_grad():
          for i, X_batch in enumerate(digit_1_loader):
            latent_vector = self.model.get_latent_var(X_batch[0].to(self.device),X_batch[1].to(self.device))
      # Сохраняем векторное пространство
      FileHandler.save_file(latent_vector, 'latent_vector')
      return latent_vector

  def get_sample(self,latent_vector,number,output_images_bool):          
      '''Создадим даталоадеры для цифр 5 и 7, которые будут состоять из полученного латентного пространства и лейба цифры'''
      label_tensor_number = torch.full((15,), number)
      number_dataloader = Data.create_data_loader(latent_vector,label_tensor_number)
      # Получаем цифру
      self.model.eval()
      with torch.no_grad():
          for i, X_batch in enumerate(number_dataloader):
            image = self.model.get_sample_var(X_batch[0].to(self.device),X_batch[1].to(self.device))
      if output_images_bool:
          Plt_show.image_show(image)
      return image
  
    