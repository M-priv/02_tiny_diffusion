import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Translates a 1D tensor of timesteps 't' into a dense matrix of sinusoidal embeddings.
    Allows the model to learn relationships across different levels of noise.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # Frequency scaling factor (constant denominator term)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Compute angles (outer product of timestep t and frequencies)
        embeddings = time[:, None] * embeddings[None, :]
        # Concatenate sine and cosine activations
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """
    Double convolution block with Group Normalization, SiLU activation,
    and direct injection of projected time embeddings.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        # Project the time embedding vector to match the number of output channels
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        # Convolutions
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # GroupNorm (splitting channels into groups is more stable than BatchNorm for small batches)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        # 1. First convolution & norm
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # 2. Project time embedding and add to spatial features
        # t_emb shape: (Batch, time_emb_dim) -> self.time_mlp(t_emb) shape: (Batch, out_ch)
        # Reshaped to (Batch, out_ch, 1, 1) to add to image height and width
        t_val = self.time_mlp(t_emb)
        h = h + t_val[:, :, None, None]
        
        # 3. Second convolution & norm
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h


class UNet(nn.Module):
    """
    Symmetric U-Net architecture designed for 28x28 grayscale images (MNIST).
    """
    def __init__(self, in_ch=1, out_ch=1, time_emb_dim=128):
        super().__init__()
        
        # 1. Time Embeddings MLP
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 2. Downward path (Encoder)
        self.init_conv = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.down1 = ConvBlock(32, 32, time_emb_dim)
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 28x28 -> 14x14
        
        self.down2 = ConvBlock(32, 64, time_emb_dim)
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 14x14 -> 7x7
        
        # 3. Bottleneck (Middle Layer)
        self.bottleneck = ConvBlock(64, 128, time_emb_dim)
        
        # 4. Upward path (Decoder)
        self.up1_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 7x7 -> 14x14
        # Input channels: 128 (upsampled) + 64 (skip connection from down2) = 192
        self.up1_block = ConvBlock(192, 64, time_emb_dim)
        
        self.up2_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 14x14 -> 28x28
        # Input channels: 64 (upsampled) + 32 (skip connection from down1) = 96
        self.up2_block = ConvBlock(96, 32, time_emb_dim)
        
        # 5. Output Conv (Maps back to target image channels)
        self.final_conv = nn.Conv2d(32, out_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        # Generate time representation
        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)
        
        # --- Downward Path (Encoder) ---
        x1 = self.init_conv(x)          # (Batch, 32, 28, 28)
        h1 = self.down1(x1, t_emb)      # (Batch, 32, 28, 28)
        p1 = self.pool1(h1)             # (Batch, 32, 14, 14)
        
        h2 = self.down2(p1, t_emb)      # (Batch, 64, 14, 14)
        p2 = self.pool2(h2)             # (Batch, 64, 7, 7)
        
        # --- Bottleneck ---
        b = self.bottleneck(p2, t_emb)  # (Batch, 128, 7, 7)
        
        # --- Upward Path (Decoder) ---
        u1 = self.up1_upsample(b)       # (Batch, 128, 14, 14)
        u1 = torch.cat([u1, h2], dim=1) # Concatenate along channel axis: shape (Batch, 192, 14, 14)
        h3 = self.up1_block(u1, t_emb)  # (Batch, 64, 14, 14)
        
        u2 = self.up2_upsample(h3)      # (Batch, 64, 28, 28)
        u2 = torch.cat([u2, h1], dim=1) # Concatenate along channel axis: shape (Batch, 96, 28, 28)
        h4 = self.up2_block(u2, t_emb)  # (Batch, 32, 28, 28)
        
        out = self.final_conv(h4)       # (Batch, out_ch, 28, 28)
        return out


if __name__ == "__main__":
    print("Running basic U-Net dimensions test...")
    model = UNet()
    
    # Create mock inputs (batch_size=8, 1 channel, 28x28 pixels)
    mock_images = torch.randn(8, 1, 28, 28)
    mock_t = torch.randint(0, 300, (8,))
    
    # Forward pass
    output = model(mock_images, mock_t)
    
    # Assert output shape matches input shape
    assert output.shape == mock_images.shape, f"Shape mismatch! Input {mock_images.shape} != Output {output.shape}"
    print(f"Test Successful! Output dimensions: {output.shape}")
