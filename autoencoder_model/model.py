import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        ''''''
        self.label_size = 10
        self.latent_dims = 4
        self.fc_dim = 3136 #7*7*64

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), #14х14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #7х7
            nn.BatchNorm2d(64),
            nn.ReLU(),   
        )
        self.flatten = nn.Flatten() # 7*7*64=3136
        self.fc = nn.Sequential( 
            nn.Linear(self.fc_dim + self.label_size, 128),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128, self.latent_dims)
        self.fc2 = nn.Linear(128, self.latent_dims)


        self.fc3 = nn.Sequential(
            nn.Linear(self.latent_dims+ self.label_size, 128),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(128, self.fc_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1,output_padding=1), #7х7
            nn.BatchNorm2d(64),
            nn.ReLU(),           
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,output_padding=1), #14х14
            nn.ReLU(),       
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1), #28х28                   
        )


    def encode(self, x, class_num):
        ''''''
        label = F.one_hot(class_num, num_classes=10)
        x_encoder = self.encoder(x)       
        x_flatten = self.flatten(x_encoder) #32, 3136
        x = self.fc(torch.cat((x_flatten, label), dim=-1)) #32, dim_code
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar, label
    
    def reparametrize(self, mu, logvar):
      '''идет после latent space'''
      if self.training:
            # <засемплируйте латентный вектор из нормального распределения с параметрами mu и sigma>
            std = logvar.mul(0.5).exp_()
            if torch.cuda.is_available():
                eps = torch.cuda.FloatTensor(std.size()).normal_()
            else:
                eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
            return eps.mul(std).add_(mu)
      else:
            # на инференсе возвращаем не случайный вектор из нормального распределения, а центральный -- mu. 
            # на инференсе выход автоэнкодера должен быть детерминирован.
            return mu
    
    def decode(self, z, class_num):
        ''''''
        x = torch.cat((z, class_num), dim=-1)
        x = self.fc3(x) #28, 3136   
        x = self.fc4(x)   
        x = x.reshape(-1, 64, 7, 7)
        x = self.output(self.decoder(x))  
        
        return torch.sigmoid(x)

    def forward(self, x, class_num):
        mu, logvar, label = self.encode(x, class_num)
        z = self.reparametrize(mu, logvar)        
        reconstruction = self.decode(z, label)       
        return mu, logvar, reconstruction

    def get_latent_var(self, x, class_num):
        ''''''
        mu, logvar, label = self.encode(x, class_num)
        z = self.reparametrize(mu, logvar)
        return z

    def get_sample_var(self, z, class_num):
        ''''''
        label = F.one_hot(class_num, num_classes=10)
        x = torch.cat((z, label), dim=-1)
        x = self.fc3(x) #28, 3136   
        x = self.fc4(x)   
        x = x.reshape(-1, 64, 7, 7)
        x = self.output(self.decoder(x))  
        return torch.sigmoid(x)