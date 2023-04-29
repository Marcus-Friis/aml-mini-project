import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow

from IPython.display import HTML
import matplotlib.animation as animation


def generate_grid(generator, noise=None, latent_dim=100, n=10, device='cpu', show=True):
    if noise is None:
        noise = torch.randn(n*n, latent_dim).to(device)
    else:
        assert noise.shape[0] == n*n, "noise must be of shape (n*n, latent_dim)"
        noise[:n]
    x_hat = generator(noise)
    x_hat = x_hat.view(-1, 28, 28)
    x_hat = x_hat.detach().cpu().numpy()
    x_hat = np.transpose(x_hat, (1, 2, 0))
    fig, ax = plt.subplots(n, n, figsize=(n, n))
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(x_hat[:, :, i*n + j], cmap='gray')
            ax[i, j].axis('off')
    if show:
        plt.show()
    return fig, ax


def training_animation(img_list, interval=1000, repeat_delay=1000, show=True):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat_delay=repeat_delay, blit=True)
    
    if show:
        HTML(ani.to_jshtml())
        
    return fig, ani


def plot_gan_loss(G_losses, D_losses, show=True):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title("Generator and Discriminator Loss During Training")
    ax.plot(G_losses,label="G")
    ax.plot(D_losses,label="D")
    ax.set_xlabel("iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    if show:
        plt.show()
    return fig, ax