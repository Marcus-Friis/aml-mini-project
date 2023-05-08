Members: Mads Høgenhaug, Marcus Friis, Morten Pedersen

<img src="figs/mnist_train_example.gif" width="400">

Central Problem: Generating synthetic drawings, based on data from Google Quickdraw. 

Domain: Generative Models

Data characteristics: 28x28 pictures. 345 different classes, with 50 million drawings in total - all drawn from hand by +15 million people. 


Central Method: Deep Convolutional Generative Adversarial Network (GAN). The generator has a latent dimension of 400 and a dropout of 0.2. It uses deconvolutional layers to generate images from noise. The discriminator is a convolutional neural network that takes an image as input and outputs a scalar value, indicating whether the input image is real or fake. 
The model uses a binary cross-entropy loss function. Both the generator and the discriminator use the Adam optimizer with a learning rate of 1e-4 and a weight decay of 1e-5. The batch size is 128, and the number of epochs is 40.

Key experiments: We construct a gif that shows how the model perfroms at each epoch, i.e. for each epoch, it shows what 

Discussion:
