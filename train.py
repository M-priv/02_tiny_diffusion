import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from unet import UNet

def get_noise_schedule(T=300, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """
    Pre-computes terms for the linear noise schedule:
    betas, alphas, and alpha_cumprod (bar_alpha).
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
    
    return sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod

def q_sample(x0, t, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod, noise=None):
    """
    Forward diffusion process:
    x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x0)
        
    sqrt_ac = sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
    sqrt_omac = sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
    
    xt = sqrt_ac * x0 + sqrt_omac * noise
    return xt, noise

def train():
    # 1. Device Selection (MPS for Mac Apple Silicon if available, else CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # 2. Hyperparameters
    T = 300
    batch_size = 64
    epochs = 5
    lr = 1e-3
    
    # 3. Noise Schedule
    sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod = get_noise_schedule(T=T, device=device)
    
    # 4. Data Loading (MNIST normalized to [-1, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 5. Model & Optimizer
    model = UNet(in_ch=1, out_ch=1, time_emb_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Starting Training Loop...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for step, (images, _) in enumerate(dataloader):
            images = images.to(device)
            current_batch_size = images.shape[0]
            
            # a. Sample a random timestep t for each image in the batch
            t = torch.randint(0, T, (current_batch_size,), device=device).long()
            
            # b. Sample random Gaussian noise epsilon
            noise = torch.randn_like(images)
            
            # c. Create noisy images x_t using closed-form forward sampling
            x_t, _ = q_sample(images, t, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod, noise=noise)
            
            # d. Forward pass through U-Net: predict the noise epsilon
            predicted_noise = model(x_t, t)
            
            # e. Compute Loss: Mean Squared Error between true noise and predicted noise
            loss = F.mse_loss(predicted_noise, noise)
            
            # f. Backpropagate & Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (step + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{step+1}/{len(dataloader)}] | MSE Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--> Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}\n")
        
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/ddpm_mnist.pt")
    print("Model checkpoint saved to checkpoints/ddpm_mnist.pt")

if __name__ == "__main__":
    train()
