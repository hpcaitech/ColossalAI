from colossalai.logging import get_dist_logger
import colossalai
import torch
import os
import torchvision
from colossalai.core import global_context as gpc
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.utils import save_image
import colossalai.nn.optimizer.fused_adam as colossalai_adam

def main():
    colossalai.launch_from_torch(config='./config.py')
    logger = get_dist_logger()
    batch_size = gpc.config.BATCH_SIZE

    mnist = MNIST(root='data',
                  train=True,
                  download=True,
                  transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))

    data_loader = DataLoader(mnist, batch_size, shuffle=True)

    image_size = 784
    hidden_size = 256

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Discriminator Network
    D = nn.Sequential(
        nn.Linear(image_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid())

    latent_size = 64

    G = nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, image_size),
        nn.Tanh())

    criterion = nn.BCELoss()

    d_optimizer = colossalai_adam(D.parameters(), lr=0.0002)

    def reset_grad():
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

    def train_discriminator(images):
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Loss for real images
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Loss for fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Combine losses
        d_loss = d_loss_real + d_loss_fake
        # Reset gradients
        reset_grad()
        # Compute gradients
        d_loss.backward()
        # Adjust the parameters using backprop
        d_optimizer.step()

        return d_loss, real_score, fake_score

    g_optimizer = colossalai_adam(G.parameters(), lr=0.0002)

    def train_generator():
        # Generate fake images and calculate loss
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        labels = torch.ones(batch_size, 1).to(device)
        g_loss = criterion(D(fake_images), labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        return g_loss, fake_images

    # Saving samples
    sample_dir = 'samples'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    sample_vectors = torch.randn(batch_size, latent_size).to(device)

    def denorm(x):
        out = (x + 1) / 2

        return out.clamp(0, 1)

    def save_fake_images(index):
        fake_images = G(sample_vectors)
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
        print('Saving', fake_fname)
        save_image(denorm(fake_images), os.path.join(
        sample_dir, fake_fname), nrow=10)

    num_epochs = gpc.config.NUM_EPOCHS
    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            # Load a batch & transform to vectors
            images = images.reshape(batch_size, -1).to(device)

            # Train the discriminator and generator
            d_loss, real_score, fake_score = train_discriminator(images)
            g_loss, fake_images = train_generator()

            # Inspect the losses
            if (i+1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))

    # Sample and save images
    save_fake_images(epoch+1)

    # Save the model checkpoints 
    torch.save(G.state_dict(), 'G.ckpt')
    torch.save(D.state_dict(), 'D.ckpt')

if __name__ == '__main__':
    main()
