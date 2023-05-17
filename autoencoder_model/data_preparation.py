import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset

class Data:
    def __init__(self):
      self.batch_size = 32
      
    def data_set():
        # MNIST Dataset
        train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)
        return train_dataset,test_dataset

    def data_loader(self):
        # Data Loader (Input Pipeline)
        train_dataset,test_dataset = Data.data_set()
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader,test_loader
    
    def create_data_loader(self,latent_vector,label_tensor_number):
        number_dataset = TensorDataset(latent_vector, label_tensor_number)
        number_dataloader = torch.utils.data.DataLoader(number_dataset, batch_size=self.batch_size, shuffle=True)
        return number_dataloader