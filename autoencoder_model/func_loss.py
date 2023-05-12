import torch
import torch.nn.functional as F

def KL_divergence(mu, logsigma):
    """
    часть функции потерь, которая отвечает за "близость" латентных представлений
    """
    loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    #loss = -0.5 * torch.sum(1 + torch.log(torch.exp(logsigma).pow(2)) - mu.pow(2) - torch.exp(logsigma).pow(2))
    #(torch.exp(sigma)**2 + mu**2 - torch.exp(sigma)) - 1/2).sum()
    return loss

def log_likelihood(x, reconstruction):
    """
    часть функции потерь, которая отвечает за качество реконструкции (как mse в обычном autoencoder)
    """
    loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')#nn.BCELoss(reduction='sum')
    #print(f"x {x.shape}")
    #print(f"reconstruction {reconstruction.shape}")
    return loss#loss(reconstruction, x)

def loss_vae(x, mu, logsigma, reconstruction):
    return KL_divergence(mu, logsigma) + log_likelihood(x, reconstruction)