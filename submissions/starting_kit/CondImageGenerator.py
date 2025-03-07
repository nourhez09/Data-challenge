# Conditional VAE Class
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim, n_layers,condition_dim=768, dropout=0.3, new_dim=768):
        super(Encoder, self).__init__()
        self.dropout = 0.1

        # MLP to process conditioning information
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, new_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(new_dim, new_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(new_dim, new_dim),
            nn.LeakyReLU(0.2)
        )

        # Convolutional layers for image encoding
        self.convs = nn.ModuleList([
            nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(hidden_dim + new_dim)
        self.fc = nn.Linear(hidden_dim + new_dim, latent_dim)

    def forward(self, x, cond):
        # Process the conditioning information
        cond_feats = self.mlp(cond)

        # Pass the image through the convolutional layers
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the pooled output (it's now (batch_size, hidden_dim, 1, 1))

        # Concatenate the conditioning features with the pooled image features
        x = torch.cat([x, cond_feats], dim=-1)

        # Batch normalization and projection to latent space
        x = self.bn(x)
        x = self.fc(x)

        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, output_channels, cond_dim=768, new_dim=768):
        super(Decoder, self).__init__()
        self.n_layers = n_layers

        # MLP to process conditioning information
        self.mlpf = nn.Sequential(
            nn.Linear(cond_dim, new_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(new_dim, new_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(new_dim, new_dim),
            nn.LeakyReLU(0.2)
        )

        # MLP layers to decode latent + condition into a feature map
        mlp_layers = [nn.Linear(latent_dim + new_dim, hidden_dim)]
        mlp_layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
        mlp_layers.append(nn.Linear(hidden_dim, 8 * 8 * hidden_dim))  # Target shape before upsampling
        self.mlp = nn.ModuleList(mlp_layers)

        self.relu = nn.ReLU()


        # Transposed convolutional layers for upsampling

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # To ensure pixel values are in the range [0, 1]
        )

    def forward(self, x, cond):
        # Process conditioning information
        cond = self.mlpf(cond)

        # Concatenate latent vector with conditioned features
        x = torch.cat([x, cond], dim=-1)

        # Pass through MLP layers
        for i in range(self.n_layers - 1):
            x = self.relu(self.mlp[i](x))

        x = self.mlp[self.n_layers - 1](x)
        # Reshape to feature map suitable for transposed convolutions
        x = x.view(x.size(0), -1, 8, 8)
        # Upsample to reconstruct the image
        x = self.deconvs(x)
        return x

class ConditionalVAE(nn.Module):
    def __init__(self, input_channels=3, hidden_dim_enc=128, hidden_dim_dec=128,
                 latent_dim=128, n_layers_enc=4, n_layers_dec=4,
                 condition_dim=768, image_size=128, cond_new_dim=768, device='cuda'):
        super(ConditionalVAE, self).__init__()

        # Initialize Encoder and Decoder
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_channels, hidden_dim_enc, latent_dim,
                               n_layers_enc, condition_dim, cond_new_dim)

        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec,
                               input_channels, condition_dim, cond_new_dim)

        # Layers to compute mu and logvar for latent space
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def encode(self, x, c):
        # Encoding step to get latent features
        x_g = self.encoder(x, c)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from latent space
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, c):
        # Decode latent vector to reconstruct the image
        return self.decoder(z, c)

    def forward(self, x, c):
        # Full forward pass through the VAE
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=0.01):
        # Compute reconstruction and KL-divergence loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kld_loss
        return loss, recon_loss, kld_loss
    
    def fit(self, dataloader):
      epochs = 1
      learning_rate = 1e-4
      beta = 0.01  # KLD loss weight

      optimizer = optim.Adam(self.parameters(), lr=learning_rate)

      # Training loop
      for epoch in range(epochs):
          self.train()
          running_loss = 0.0

          for i, (condition, image) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
              condition = condition.to(device)  # Text embeddings
              image = image.to(device)          # Image tensor
              optimizer.zero_grad()

              # Forward pass
              recon_image, mu, logvar = self.forward(image, condition)

              # Compute the loss
              loss, recon_loss, kld_loss = self.loss_function(recon_image, image, mu, logvar, beta)

              # Backpropagation
              loss.backward()
              optimizer.step()

              running_loss += loss.item()

          avg_loss = running_loss / len(dataloader)
          print(f"Loss: {avg_loss:.4f}")
          # Save the model after every epoch
          torch.save(self.state_dict(), f"vae_epoch_{epoch+1}.pth")

    @torch.no_grad()
    def predict(self, dataloader):
      self.eval()
      res = []
      for i, (condition, _) in enumerate(tqdm(dataloader)):
        condition = condition.to(device)
        n = len(condition)
        samples = torch.randn((n, self.latent_dim), device=condition.device)
        # print(samples.shape, condition.shape)
        decoded_samples = self.decode(samples, condition)
        res.append(decoded_samples.cpu().numpy())
      return np.concat(res)
