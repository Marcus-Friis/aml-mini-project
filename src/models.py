import torch
import torch.nn as nn

class SuperDeepGenerator(nn.Module):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) generator,
    Generates images from a random noise vector (latent_dim, 1, 1).
    """
    def __init__(self,
                latent_dim: int = 100,
                dropout: float = 0.2,
                ngf: int = 64
                ):
        super().__init__()

        self.latent_dim = latent_dim
        self.dropout = dropout
        self.ngf = ngf

        self.main = nn.Sequential(
            # latent_dim, 1, 1
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            # ngf * 4, 4, 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            # ngf * 2, 8, 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Dropout(dropout),
            # ngf, 14, 14
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # 1, 28, 28
        )

    def forward(self, input):
        return self.main(input)


class SuperDeepConvDiscriminator(nn.Module):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) discriminator,
    Discriminates between real and generated images.
    """
    def __init__(self,
                 ndf: int = 64):
        super(SuperDeepConvDiscriminator, self).__init__()

        self.ndf = ndf

        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf*8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(ndf*8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


##########################################
# Models beneath this line are not used. #
##########################################


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
    latent_dim = 100
    x = torch.randn(64, latent_dim, 1, 1)
    model = SuperDeepGenerator()
    print(model(x).shape)

    x = torch.randn(64, 1, 28, 28)
    model = SuperDeepConvDiscriminator()
    print(model(x).shape)