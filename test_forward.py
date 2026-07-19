import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def get_noise_schedule(T=300, beta_start=1e-4, beta_end=0.02):
    """
    Computes a linear schedule for adding noise.
    """
    # 1. Linear beta schedule: variance added at each step
    betas = torch.linspace(beta_start, beta_end, T)
    
    # 2. alphas = 1 - beta: proportion of original image kept at each step
    alphas = 1.0 - betas
    
    # 3. alpha_cumprod (bar_alpha): cumulative product of alphas
    # This represents the total signal remaining from x0 up to step t
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    # Pre-calculate sqrt terms for quick sampling
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
    
    return sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod

def q_sample(x0, t, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
    """
    Forward diffusion process: Closed-form sampling of x_t given x_0 and t.
    x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * noise
    """
    noise = torch.randn_like(x0)
    
    # Retrieve the coefficients for the given timesteps
    # We reshape to (batch_size, 1, 1, 1) so it broadcasts over image channels and dimensions
    sqrt_ac = sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
    sqrt_omac = sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
    
    xt = sqrt_ac * x0 + sqrt_omac * noise
    return xt, noise

def main():
    # 1. Set up data transforms (Normalize images to [-1, 1] for stable GAN/Diffusion training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 2. Download and load MNIST dataset
    print("Downloading/Loading MNIST dataset...")
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Get a batch of 5 images
    loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    images, labels = next(iter(loader))
    
    # 3. Compute the noise schedule
    T = 300
    sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod = get_noise_schedule(T)
    
    # 4. Generate noisy versions at different timesteps
    timesteps_to_show = [0, 50, 100, 150, 200, 250, 299]
    fig, axes = plt.subplots(len(images), len(timesteps_to_show), figsize=(12, 8))
    
    for i in range(len(images)):
        x0 = images[i:i+1] # Keep batch dimension -> shape (1, 1, 28, 28)
        
        for col_idx, t_val in enumerate(timesteps_to_show):
            t = torch.tensor([t_val])
            
            if t_val == 0:
                xt = x0
            else:
                xt, _ = q_sample(x0, t, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod)
                
            # Denormalize image from [-1, 1] back to [0, 1] for visualization
            img_to_show = (xt[0, 0].numpy() + 1.0) / 2.0
            img_to_show = np.clip(img_to_show, 0, 1)
            
            ax = axes[i, col_idx]
            ax.imshow(img_to_show, cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f"t = {t_val}")
                
    plt.tight_layout()
    save_path = "/Users/michaeladesiyan/.gemini/antigravity/brain/b02b9c28-62e5-47dc-8f71-3493715d3199/diffusion_forward_process.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Forward diffusion visualization saved successfully to: {save_path}")

if __name__ == "__main__":
    main()
