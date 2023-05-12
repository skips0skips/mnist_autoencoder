import numpy as np
import torch
import matplotlib.pyplot as plt

from autoencoder_model.data_preparation import data_loader

from autoencoder_model.func_loss import loss_vae

from autoencoder_model.model import CVAE



device = torch.device('cpu')#'cuda:0' if torch.cuda.is_available() else 'cpu'

criterion = loss_vae
model = CVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

train_loader,test_loader = data_loader()

X_val = next(iter(test_loader))

# Вывод изображений при обучении
output_images_bool = False

# Проверка на валидационной выболрке
val_fit_bool = True



def train(output_images_bool, val_fit_bool, n_epochs=50, ):
    
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):

        avg_loss = 0

        model.train()
        train_losses_per_epoch = []

        for i, X_batch in enumerate(train_loader):
            optimizer.zero_grad()
            mu, log_var, reconstruction = model(X_batch[0].to(device),X_batch[1].to(device))
            loss = criterion(X_batch[0].to(device).float(), mu, log_var, reconstruction) #((X_batch[0].to(device) - reconstruction)**2).sum() + (log_var**2 + mu**2 - torch.log(log_var) - 1/2).sum()
            loss.backward()
            optimizer.step()
            train_losses_per_epoch.append(loss.item())
            avg_loss += loss / len(train_loader) #рассчитайте потери, чтобы показать пользователю
        train_losses.append(np.mean(train_losses_per_epoch))#значения для вывода графика лосса

        if output_images_bool:
            plt_show(reconstruction, epoch, avg_loss, n_epochs)
        else:
            print('%d / %d - loss: %f' % (epoch+1, n_epochs, avg_loss))

        if val_fit_bool:

            model.eval()
            val_losses_per_epoch = []
            #clear_output(wait=True)#очищаем прошлые изображения
            with torch.no_grad():
                mu, log_var, reconstruction = model(X_val[0].to(device),X_val[1].to(device))
                loss = criterion(X_val[0].to(device).float(), mu, log_var, reconstruction) #((X_val[0].to(device) - reconstruction)**2).sum() + (log_var**2 + mu**2 - torch.log(log_var) - 1/2).sum()
                val_losses_per_epoch.append(loss.item())

            val_losses.append(np.mean(val_losses_per_epoch))            

def plt_show(reconstruction, epoch, avg_loss, n_epochs):
    '''
    '''
    fig, axs = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    for i in range(10):
        axs[0][i].imshow(X_val[0][i].permute(1, 2, 0).cpu(), cmap='gist_gray')
        axs[0][i].axis('off')
        axs[0][i].set_title('X {}'.format(i))
        axs[1][i].imshow(reconstruction[i].permute(1, 2, 0).cpu(), cmap='gist_gray')
        axs[1][i].axis('off')
        axs[1][i].set_title('reconstructed {}'.format(i))
    plt.suptitle('%d / %d - loss: %f' % (epoch+1, n_epochs, avg_loss))
    plt.show()