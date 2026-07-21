# Tiny Diffusion Model (DDPM)

A minimal, fully functional PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) trained from scratch to generate 28x28 handwritten digits (MNIST).

## 📁 Repository Structure

*   `test_forward.py`: Implements and visualizes the forward diffusion noise process ($q(x_t | x_0)$) across timesteps.
*   `unet.py`: The core U-Net neural network architecture with Sinusoidal Time Embeddings and skip connections.
*   `train.py`: Training script executing the linear variance schedule, MSE noise prediction loss, and saving model checkpoints.
*   `sample.py`: Iterative reverse denoising sampler ($p_\theta(x_{t-1} | x_t)$) that generates brand-new images from pure Gaussian noise.

---

## 🚀 Quick Start

### 1. Installation & Environment Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib
```

### 2. Forward Diffusion Visualization
Inspect how Gaussian noise corrupts clean images over 300 timesteps:
```bash
python test_forward.py
```

### 3. Train the Model
Train the U-Net to predict noise on MNIST (automatically uses Apple Silicon MPS GPU if available):
```bash
python train.py
```

### 4. Generate Digits (Sampling)
Generate brand-new handwritten digits from pure noise:
```bash
python sample.py
```

---

## 🧑‍🔬 Mathematical Highlights

### 1. Closed-Form Forward Process
Directly sample $x_t$ at any timestep $t$ without looping:
$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}) $$

### 2. Noise Prediction Objective
Instead of predicting $x_0$ directly, the model minimizes the MSE between actual noise $\epsilon$ and predicted noise $\hat{\epsilon}$:
$$ \mathcal{L} = \| \epsilon - \text{UNet}(x_t, t) \|^2 $$

### 3. Reverse Denoising Sampling (Algorithm 2)
Iteratively subtract predicted noise from $t = T-1$ down to $0$:
$$ x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \text{UNet}(x_t, t) \right) + \sqrt{\beta_t} z $$

---

## 🎨 Results

Generated handwritten digits created out of thin air by our DDPM model:

![Generated MNIST Digits](/Users/michaeladesiyan/.gemini/antigravity/brain/b02b9c28-62e5-47dc-8f71-3493715d3199/generated_mnist_digits.png)
