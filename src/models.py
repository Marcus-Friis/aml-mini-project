import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 latent_dim: int = 100,
                 dropout: float = 0.2
                 ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.dropout = dropout
        
        self.sequence = nn.Sequential(
            # Linear
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3 * 3 * 32),
            nn.GELU(),
            nn.Dropout(dropout),
            # Reshape
            nn.Unflatten(1, (32, 3, 3)),
            # Convolve
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.sequence(x)
    
    
class DeepGenerator(nn.Module):
    def __init__(self,
                 latent_dim: int = 100,
                 dropout: float = 0.2
                 ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.dropout = dropout
        
        self.sequence = nn.Sequential(
            # Linear
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3 * 3 * 128),
            nn.GELU(),
            nn.Dropout(dropout),
            # Reshape
            nn.Unflatten(1, (128, 3, 3)),
            # Convolve
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.sequence(x)    

    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.sequence(x)
    
class ConvDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=10, out_channels=30, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(120, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.sequence(x)
    
    
if __name__ == '__main__':
    x = torch.randn(64, 1, 28, 28)
    
    latent_dim = 100
    noise = torch.randn(64, latent_dim)
    model = ConvDiscriminator()
    
    print(model(x).shape)