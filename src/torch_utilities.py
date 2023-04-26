import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from IPython.display import HTML
import matplotlib.animation as animation


def generate_grid(generator, latent_dim=100, n=10, device='cpu'):
    noise = torch.randn(n*n, latent_dim).to(device)
    x_hat = generator(noise)
    x_hat = x_hat.view(-1, 28, 28)
    x_hat = x_hat.detach().cpu().numpy()
    x_hat = np.transpose(x_hat, (1, 2, 0))
    fig, ax = plt.subplots(n, n, figsize=(n, n))
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(x_hat[:, :, i*n + j], cmap='gray')
            ax[i, j].axis('off')
    plt.show()
    return fig, ax


def training_animation(img_list, interval=1000, repeat_delay=1000):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat_delay=repeat_delay, blit=True)

    HTML(ani.to_jshtml())
    return fig, ani


# write boilerplate code using wandb for logging a models performance
def wandb_init(project_name, model, config, notes=None):
    wandb.init(project=project_name, config=config, notes=notes)
    wandb.watch(model)