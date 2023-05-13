import numpy as np
import torch

from autoencoder_model.config import Config

from autoencoder_model.data_preparation import data_loader

from autoencoder_model.file_handler import FileHandler

from autoencoder_model.plt_show import Plt_show


class Trainer:
    def __init__(self):
        self.model, self.criterion, self.optimizer, self.val_fit_bool, self.output_images_bool, self.device = Config.get_config()
        self.test_loader, self.train_loader = data_loader()

    def train(self, output_images_bool, val_fit_bool, n_epochs=50):
        
        if val_fit_bool:
            X_val = next(iter(self.test_loader))

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):

            avg_loss = 0

            self.model.train()
            train_losses_per_epoch = []

            for i, X_batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                mu, log_var, reconstruction = self.model(X_batch[0].to(self.device),X_batch[1].to(self.device))
                loss = self.criterion(X_batch[0].to(self.device).float(), mu, log_var, reconstruction) 
                loss.backward()
                self.optimizer.step()
                train_losses_per_epoch.append(loss.item())
                avg_loss += loss / len(self.train_loader) #рассчитайте потери, чтобы показать пользователю
            train_losses.append(np.mean(train_losses_per_epoch))#значения для вывода графика лосса
            
            print('%d / %d - loss: %f' % (epoch+1, n_epochs, avg_loss))
            

            if val_fit_bool:

                self.model.eval()
                val_losses_per_epoch = []
                with torch.no_grad():
                    mu, log_var, reconstruction = self.model(X_val[0].to(self.device),X_val[1].to(self.device))
                    loss = self.criterion(X_val[0].to(self.device).float(), mu, log_var, reconstruction) #((X_val[0].to(device) - reconstruction)**2).sum() + (log_var**2 + mu**2 - torch.log(log_var) - 1/2).sum()
                    val_losses_per_epoch.append(loss.item())

                val_losses.append(np.mean(val_losses_per_epoch))

                if output_images_bool:
                    Plt_show.plt_show(reconstruction, epoch, avg_loss, n_epochs,X_val)
        
        # Сохраняем self.model
        FileHandler.save_file(self.model, 'self.model')

        # Сохраняем Loss
        FileHandler.save_file(train_losses, 'train_losses')      
        return self.model 
               