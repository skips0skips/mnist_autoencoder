import numpy as np
import torch


def get_latent_vector():

    '''Возьмем к примеру изображение единицы и получим латентный вектор'''
    # Индекс цифры 1 в датасете
    digit_1_indices = torch.where(test_dataset.targets == 1)[0]

    # Создание подмножества из датасета train_dataset с индексами digit_1_indices
    digit_1_subset = Subset(test_dataset, digit_1_indices)

    # Создание DataLoader для подмножества digit_1_subset
    digit_1_loader = torch.utils.data.DataLoader(dataset=digit_1_subset, batch_size=batch_size, shuffle=True)

    '''Получим латентный вектор цифры 1'''
    model.eval()
    with torch.no_grad():
        for i, X_batch in enumerate(digit_1_loader):
        latent_one = model.get_latent_var(X_batch[0].to(device),X_batch[1].to(device))

'''Создадим даталоадеры для цифр 5 и 7, которые будут состоять из полученного латентного пространства и лейба цифры'''
label_tensor_seven = torch.full((15,), 7)
dataset_seven = TensorDataset(latent_one, label_tensor_seven)
dataloader_seven = DataLoader(dataset_seven, batch_size=batch_size, shuffle=True)

label_tensor_five = torch.full((15,), 5)
dataset_five = TensorDataset(latent_one, label_tensor_five)
dataloader_five = DataLoader(dataset_five, batch_size=batch_size, shuffle=True)

'''Получаем цифру 7'''
model.eval()
with torch.no_grad():
    for i, X_batch in enumerate(dataloader_seven):
      image_seven = model.get_sample_var(X_batch[0].to(device),X_batch[1].to(device))
plt.figure(figsize=(50, 50))
for i, img in enumerate(image_seven):
  plt.subplot(1, 15, i+1)
  plt.imshow(img.permute(1, 2, 0).cpu().detach().numpy(), cmap='gist_gray')
  