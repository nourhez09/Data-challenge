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


class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim, n_layers, condition_dim=768, dropout=0.3, new_dim=768, device='cuda'):
        super(Encoder, self).__init__()
        self.dropout = 0.1
        self.device = device
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.new_dim = new_dim
        # MLP to process conditioning information
        self.mlp = nn.Sequential(
            nn.Linear(self.condition_dim, self.new_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.new_dim, self.new_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.new_dim, self.new_dim),
            nn.LeakyReLU(0.2)
        ).to(self.device)

        # Convolutional layers for image encoding
        self.convs = nn.ModuleList([
            nn.Conv2d(self.input_channels, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1)
        ]).to(self.device)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1).to(self.device)
        self.bn = nn.BatchNorm1d(hidden_dim + self.new_dim).to(self.device)
        self.fc = nn.Linear(self.hidden_dim + self.new_dim, self.latent_dim).to(self.device)

    def forward(self, x, cond):
        x = x.to(self.device)
        cond = cond.to(self.device)
        
        # Process the conditioning information
        cond_feats = self.mlp(cond)
        cond_feats = cond_feats.to(device=self.device)

        # Pass the image through the convolutional layers
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the pooled output

        # Concatenate the conditioning features with the pooled image features
        x = torch.cat([x, cond_feats], dim=-1)

        # Batch normalization and projection to latent space
        x = self.bn(x)
        x = self.fc(x)

        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, output_channels, cond_dim=768, new_dim=768, device='cuda'):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.device = device
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.cond_dim = cond_dim
        self.new_dim = new_dim
        self.n_layers = n_layers

        # MLP to process conditioning information
        self.mlpf = nn.Sequential(
            nn.Linear(self.cond_dim, self.new_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.new_dim, self.new_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.new_dim, self.new_dim),
            nn.LeakyReLU(0.2)
        ).to(self.device)

        # MLP layers to decode latent + condition into a feature map
        mlp_layers = [nn.Linear(self.latent_dim + self.new_dim, self.hidden_dim)]
        mlp_layers += [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layers - 2)]
        mlp_layers.append(nn.Linear(self.hidden_dim, 8 * 8 * self.hidden_dim))
        self.mlp = nn.ModuleList(mlp_layers).to(self.device)

        self.relu = nn.ReLU().to(self.device)

        # Transposed convolutional layers for upsampling
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.hidden_dim, self.output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, x, cond):
        x = x.to(self.device)
        cond = cond.to(self.device)

        cond = self.mlpf(cond)
        x = torch.cat([x, cond], dim=-1)

        for i in range(self.n_layers - 1):
            x = self.relu(self.mlp[i](x))
        x = self.mlp[self.n_layers - 1](x)

        x = x.view(x.size(0), -1, 8, 8)
        x = self.deconvs(x)
        return x


class ConditionalVAE(nn.Module):
    def __init__(self, input_channels=3, hidden_dim_enc=128, hidden_dim_dec=128,
                 latent_dim=128, n_layers_enc=4, n_layers_dec=4,
                 condition_dim=768, image_size=128, cond_new_dim=768, device='cuda'):
        super(ConditionalVAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.cond_new_dim = cond_new_dim
        self.hidden_dim_enc = hidden_dim_enc
        self.hidden_dim_dec = hidden_dim_dec
        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec
        self.encoder = Encoder(input_channels=self.input_channels, hidden_dim=self.hidden_dim_enc, latent_dim=self.latent_dim, n_layers=self.n_layers_enc, condition_dim=self.condition_dim, new_dim=self.cond_new_dim, device=self.device)
        self.decoder = Decoder(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim_dec, n_layers=self.n_layers_dec, output_channels=self.input_channels, cond_dim=self.condition_dim, new_dim=self.cond_new_dim, device=self.device)

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)

    def encode(self, x, c):
        x, c = x.to(self.device), c.to(self.device)
        x_g = self.encoder(x, c)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(self.device)
            return mu + eps * std
        return mu

    def decode(self, z, c):
        return self.decoder(z.to(self.device), c.to(self.device))

    def forward(self, x, c):
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
              condition = condition.to(self.device)  # Text embeddings
              image = image.to(self.device)          # Image tensor
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
        for condition, _ in tqdm(dataloader):
            condition = condition.to(self.device)
            n = len(condition)
            samples = torch.randn((n, self.latent_dim), device=self.device)
            decoded_samples = self.decode(samples, condition)
            res.append(decoded_samples.cpu().numpy())
        return np.concatenate(res)