import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # Output: (16, 64, 80)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: (32, 32, 40)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: (64, 16, 20)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*16*20, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Latent space (10 for mu, 10 for logvar)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64*16*20),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 20)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output: (1, 128, 160)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = encoded[:, :2], encoded[:, 2:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

class FrameDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0) / 255.0
        return frame


def visualize_reconstruction(original, reconstructed, epoch, n=10):
    """
    Visualizes original and reconstructed images.
    :param original: tensor of original images
    :param reconstructed: tensor of reconstructed images
    :param epoch: current epoch number
    :param n: number of images to display
    """
    original = original[:n].cpu().detach().numpy()
    reconstructed = reconstructed[:n].cpu().detach().numpy()
    
    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        # Display original images
        axes[0, i].imshow(original[i].reshape(128, 160), cmap='gray')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].set_title(f'Original {i+1}')
        # Display reconstructed images
        axes[1, i].imshow(reconstructed[i].reshape(128, 160), cmap='gray')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].set_title(f'Reconstructed {i+1}')
    plt.show()

def train_vae(dataset, epochs=1000, batch_size=32, learning_rate=1e-3, visualize_every=10, save_every=10, save_dir='./'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create directory for saving models if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
        # Visualization at specified intervals
        if (epoch + 1) % visualize_every == 0:
            with torch.no_grad():
                model.eval()
                sample = next(iter(dataloader))[:10].to(device)
                reconstructed, _, _ = model(sample)
                visualize_reconstruction(sample, reconstructed, epoch+1)
                model.train()
        
        # Save model at specified intervals
        if (epoch + 1) % save_every == 0:
            save_path = os.path.join(save_dir, f'vae_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')


# Load dataset and train
dataset = FrameDataset('train.npy')
train_vae(dataset)

