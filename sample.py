import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet

def get_noise_schedule(T=300, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """
    Computes betas, alphas, and cumulative alphas for the reverse denoising steps.
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
    
    return betas, alphas, sqrt_recip_alphas, sqrt_one_minus_alpha_cumprod

@torch.no_grad()
def p_sample_loop(model, shape, T, betas, sqrt_recip_alphas, sqrt_one_minus_alpha_cumprod, device):
    """
    Reverse Denoising Sampling Loop (DDPM Algorithm 2):
    Starts from pure Gaussian noise at t = T-1 and iteratively denoises down to t = 0.
    """
    model.eval()
    
    # 1. Start with pure random static x_T ~ N(0, I)
    x = torch.randn(shape, device=device)
    
    print("Starting reverse denoising sampling loop...")
    for t in reversed(range(T)):
        # Create timestep tensor for the current batch
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Predict the noise using U-Net
        predicted_noise = model(x, t_batch)
        
        # Pre-calculated coefficients for step t
        beta_t = betas[t]
        sqrt_recip_alpha_t = sqrt_recip_alphas[t]
        sqrt_omac_t = sqrt_one_minus_alpha_cumprod[t]
        
        # Calculate mean for x_{t-1}
        # mean = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - bar_alpha_t)) * predicted_noise)
        model_mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_omac_t) * predicted_noise)
        
        if t > 0:
            # Add stochastic noise z ~ N(0, I) scaled by sqrt(beta_t)
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = model_mean + sigma_t * noise
        else:
            x = model_mean
            
        if (t + 1) % 50 == 0 or t == 0:
            print(f"Denoising Step [{T - t}/{T}] (t = {t}) complete.")
            
    return x

def main():
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Sampling on device: {device}")
    
    T = 300
    num_samples = 16
    
    # 1. Compute noise schedule terms
    betas, alphas, sqrt_recip_alphas, sqrt_one_minus_alpha_cumprod = get_noise_schedule(T=T, device=device)
    
    # 2. Load trained U-Net model checkpoint
    model = UNet(in_ch=1, out_ch=1, time_emb_dim=128).to(device)
    checkpoint_path = "checkpoints/ddpm_mnist.pt"
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with randomly initialized model.")
        
    # 3. Generate samples
    samples = p_sample_loop(
        model, 
        shape=(num_samples, 1, 28, 28), 
        T=T, 
        betas=betas, 
        sqrt_recip_alphas=sqrt_recip_alphas, 
        sqrt_one_minus_alpha_cumprod=sqrt_one_minus_alpha_cumprod, 
        device=device
    )
    
    # 4. Save generated digits grid
    samples = samples.cpu()
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)
    
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0].numpy(), cmap="gray")
        ax.axis("off")
        
    plt.suptitle("Generated MNIST Digits (DDPM)", fontsize=14)
    plt.tight_layout()
    
    save_path = "/Users/michaeladesiyan/.gemini/antigravity/brain/b02b9c28-62e5-47dc-8f71-3493715d3199/generated_mnist_digits.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Generated digits successfully saved to artifact: {save_path}")

if __name__ == "__main__":
    main()
