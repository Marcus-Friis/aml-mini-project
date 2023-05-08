import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from tqdm import tqdm

from src.torch_utilities import *
from src.models import *

from IPython.display import HTML
import matplotlib.animation as animation

import warnings
warnings.filterwarnings('ignore')


def get_mnist():
    trainset = datasets.MNIST('data', download=True, train=True, transform=transform)
    valset = datasets.MNIST('data', download=True, train=False, transform=transform)
    return trainset  #, valset


def get_cats():
    img_dim = 28
    cats = np.load('data/cats/full_numpy_bitmap_cat.npy').reshape(-1, 1, img_dim, img_dim) / 255
    n_cats = cats.shape[0]

    x = cats  
    y = np.ones(n_cats)

    trainset = TensorDataset(torch.tensor(x).float(), torch.tensor(y).float())
    return trainset

if __name__ == '__main__':
    # run parameters
    verbose = True
    mlflow_log = True

    # setup device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('using device: ', device)

    # hyperparameters
    batch_size = 64
    ## define model hyperparameters
    latent_dim = 100
    dropout = 0.3
    ## setup training hyperparameters
    lr = 1e-3
    n_epochs = 50

    # instantiate model
    generator = DeepGenerator(latent_dim=latent_dim, dropout=dropout).to(device)
    discriminator = ConvDiscriminator().to(device)
    
    # setup mlflow tracking
    mlflow.set_tracking_uri('http://localhost:5000')
    
    # transform input image
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    # setup data
    trainset = get_cats()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    
    # generate fixed noise for visualization of training progress
    fixed_noise = torch.randn(64, latent_dim, device=device)

    # setup real and fake labels for loss function
    real_label = 1.
    fake_label = 0.

    # setup loss function
    loss_fn = nn.BCELoss()

    # setup optimizers
    g_optim = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)

    # setup lists for tracking training progress
    img_list = []
    G_losses = []
    D_losses = []
    G_batch_losses = []
    D_batch_losses = []
    iters = 0
    
    try:
        for epoch in range(1, n_epochs + 1):
            for i, (real, _) in enumerate(trainloader):
                ## train the discriminator on real data
                # make the discriminator predict on the real data
                real = real.to(device)
                b_size = real.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output = discriminator(real).view(-1)
                
                # calculate the loss
                d_loss_real = loss_fn(output, label)
                
                ## train the discriminator on fake data
                # generate fake data
                noise = torch.randn(b_size, latent_dim, device=device)
                fake = generator(noise)
                label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
                
                # make the discriminator predict on the fake data
                output = discriminator(fake.detach()).view(-1)
                
                # calculate the loss
                d_loss_fake = loss_fn(output, label)
        
                # compute full loss and backpropagate       
                d_loss = d_loss_real + d_loss_fake
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()
                
                
                ## train the generator
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                
                # make the discriminator predict on the fake data
                output = discriminator(fake).view(-1)
                
                # calculate the loss for the generator
                g_loss = loss_fn(output, label)
                
                # backpropagate
                g_optim.zero_grad()
                g_loss.backward()
                g_optim.step()
                
                # Save Losses for plotting later
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
                    
                # print training progress
                if verbose and i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch, n_epochs, i, len(trainloader),
                            d_loss.item(), g_loss.item()))
                    
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == n_epochs-1) and (i == len(trainloader)-1)):
                    with torch.no_grad():
                        fake = generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                # increment iteration counter
                iters += 1
                
            # save batch losses
            G_batch_losses.append(np.mean(G_losses[-b_size:]))
            D_batch_losses.append(np.mean(D_losses[-b_size:]))
    except KeyboardInterrupt:
        print('Interrupted training')

    print('logging to mlflow...')
    if mlflow_log:
        
        run_description = f'trained on MNIST for {n_epochs} epochs with batch size {batch_size}, models:\n{generator.__repr__()} \n{discriminator.__repr__()}'
        
        # save run and model to mlflow
        with mlflow.start_run(description=run_description):
            # log parameters
            params = {
                # training parameters
                'n_epochs': epoch,
                'lr': lr,
                'latent_dim': latent_dim,
                'batch_size': batch_size,
                'optimizer': d_optim.__repr__(),
                'loss_fn': loss_fn.__repr__(),
                
                # model parameters
                'latent_dim': latent_dim,
                'dropout': dropout,
            }

            mlflow.log_params(params)
            
            # log metrics
            for i in range(len(G_batch_losses)):
                metrics = {
                    'g_loss': G_batch_losses[i],
                    'd_loss': D_batch_losses[i]
                }
                
                mlflow.log_metrics(metrics, step=i)
            
            # log loss curves
            fig, ax = plot_gan_loss(G_batch_losses, D_batch_losses, show=False)
            mlflow.log_figure(fig, 'gan_loss.svg')
            plt.close(fig)
            
            # log training animation
            for i, img in enumerate(img_list):
                fig, ax = plt.subplots()
                ax.imshow(np.transpose(img,(1,2,0)), animated=True)
                mlflow.log_figure(fig, f'training_{i}.svg')
                plt.close(fig)
                
            # log model
            mlflow.pytorch.log_model(generator, generator.__class__.__name__)
            mlflow.pytorch.log_model(discriminator, generator.__class__.__name__)
            
            # gif test
            try:
                mlflow.log_artifact('kitty-cat-sandwich.gif')
                fig, ani = training_animation(img_list, save=True)
                mlflow.log_artifact('training.mp4')
            except:
                print('failed to log animation')
