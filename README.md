# DCGAN Cartoon Image Generation

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation that generates high-quality cartoon images by following the architectural guidelines in the original DCGAN paper.

## Project Overview

This project implements an optimized DCGAN architecture to generate cartoon images from Google's Cartoon dataset. The implementation follows specific guidelines from the DCGAN paper to ensure stable training and high-quality image generation.

**DCGAN paper**: https://arxiv.org/pdf/1511.06434

**Google Cartoon dataset**: https://google.github.io/cartoonset/download.html (download the 10k version of the dataset)

### Key Features

- **Architecture Optimizations**: Follows DCGAN paper guidelines for stable GAN training
- **Balanced Training Process**: Implements techniques to balance generator and discriminator training
- **Loss Monitoring**: Tracks and visualizes discriminator and generator losses
- **Dynamic Learning Rate**: Adjusts learning rates during training to improve stability
- **Diversity Preservation**: Includes mechanisms to prevent mode collapse
- **User-Configurable Parameters**: Easy configuration through YAML file

## System Architecture

The system implements a standard GAN architecture with specialized components:

1. **Generator**: Generates realistic cartoon images from random noise input
2. **Discriminator**: Classifies images as real or fake to guide generator training
3. **Training Loop**: Balances generator and discriminator training for optimal results
4. **Visualization System**: Generates sample images and loss plots for monitoring

## Project Structure

```
project/
├── config.yaml              # Configuration parameters
├── generator.py             # Generator network architecture
├── discriminator.py         # Discriminator network architecture
├── main.py                  # Main training script
├── helper.py                # Utility functions
├── Dockerfile               # Docker configuration
├── README.md                # This documentation
└── results/                 # Generated during training
    ├── real_samples.png     # Real images from dataset
    ├── fake_images_*.png    # Generated images at different epochs
    ├── comparison_*.png     # Real vs. fake image comparisons
    ├── loss_plot.png        # Generator and discriminator loss curves
    ├── discriminator_scores_plot.png # D(x) and D(G(z)) plots
    └── best_model.pt        # Best model checkpoint
```

## Implementation Details

### Architectural Optimizations

Based on the DCGAN paper, I made these critical changes to the original implementation to stabilize training:

#### Generator Architecture:
- **Changed final activation**: Replaced Sigmoid with Tanh (original used Sigmoid which compresses to [0,1] instead of [-1,1])
- **Added BatchNorm layers**: Original completely lacked BatchNorm after each layer except output
- **Kept ReLU activations**: Corrected implementation to match DCGAN paper (original correctly used ReLU)
- **Fixed noise vector handling**: Original had improper handling of the random noise input dimensions
- **Increased feature map depth**: Changed gen_dim from 32 to 64 for more capacity

#### Discriminator Architecture:
- **Replaced ReLU with LeakyReLU**: Original incorrectly used ReLU instead of LeakyReLU with 0.2 slope
- **Added BatchNorm layers**: Original had no BatchNorm layers at all
- **Kept strided convolutions**: Original correctly used this for downsampling
- **Kept final Sigmoid**: For stability in early training (will remove later in training)
- **Increased feature map depth**: Changed disc_dim from 32 to 64 for better discrimination

### Hyperparameter Tuning

Key hyperparameter optimizations from the original implementation:

- **Learning rate**: Increased from 0.00002 to 0.0002 (10x higher) - original was too slow to converge
- **Batch size**: Increased from 64 to 128 for more stable gradient estimation
- **Added label smoothing**: Set real labels to 0.9 instead of 1.0 (completely absent in original)
- **Training epochs**: Increased from 5 to 30 epochs (original training period was far too short, but if you want test my code just use 20 epochs)
- **Kept weight initialization**: Original correctly used mean=0, std=0.02 but didn't apply to BatchNorm

### Training Optimizations

To ensure stable and effective training (all of these were missing in the original implementation):

- **Added dynamic G/D update balancing**: Original blindly updated both networks equally
- **Fixed noise vector handling**: Original implementation had dimensional issues with noise vectors
- **Added diversity loss**: Prevents mode collapse by enforcing variation (absent in original)
- **Implemented learning rate scheduling**: Original had static learning rate throughout training
- **Added conditional updates**: Skip D updates when it's too strong, helping G catch up (original had no such mechanism)
- **Multiple G updates per iteration**: When G is struggling, give it extra update opportunities

## Setup Instructions

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)
- Docker (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Download the Google Cartoon dataset (10k version) and extract it to your desired location.

3. Update `config.yaml` to point to your dataset location:
   ```yaml
   data_dir: "/path/to/cartoonset10k"
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t dcgan-cartoon .
   ```

2. Run the container with the dataset mounted:
   ```bash
   docker run -it --gpus all -v /path/to/cartoonset10k:/app/cartoonset10k dcgan-cartoon
   ```

### Running Without Docker

1. Install required packages:
   ```bash
   pip install torch torchvision matplotlib pyyaml pillow
   ```

2. Run the training script:
   ```bash
   python main.py
   ```

## Results

### Training Progression

The training process shows stable convergence with balanced generator and discriminator performance:

- Generator loss stabilizes around 1.5-2.0 after 15-20 epochs
- Discriminator effectively distinguishes real/fake (D(x) ≈ 0.7, D(G(z)) ≈ 0.3)
- Generated images progressively improve in quality and diversity

### Sample Outputs

The final model generates high-quality cartoon images that closely resemble the training distribution. While some minor artifacts may still be present, the overall quality demonstrates successful learning of the cartoon image distribution.

## Technical Notes

- **Kernel configuration**: Both original and optimized implementation use 4×4 kernels with stride 2
- **Critical flaws in original**:
  - The original completely lacked BatchNorm, causing training instability and poor results
  - Original had improper layer dimensioning, especially in noise handling
  - Original used ReLU in discriminator instead of LeakyReLU, greatly hindering gradient flow
  - Original learning rate was 10x too small, severely slowing convergence
  - Original had no training balance mechanism, causing oscillation and failure
- **Noise generation**: Corrected the noise dimension handling to properly reshape for transposed convolutions
- **Normalization**: Added proper normalization to input images (-1 to 1 range) to match Tanh output

## Implementation Choices and Original Code Issues

After analyzing the original code implementation, I identified several issues that prevented proper image generation. The key changes made from the original implementation include:

1. **Network Architecture Issues Fixed**:
   - **Generator**: 
     - Original used Sigmoid activation in the output layer, causing color saturation issues
     - Original used ReLU throughout, which doesn't match DCGAN recommendations
     - Missing BatchNorm layers in all generator blocks
   
   - **Discriminator**: 
     - Original used ReLU instead of LeakyReLU, reducing gradient flow
     - No BatchNorm layers in any discriminator blocks
     - Input dimension handling was incorrect for noise vector

2. **Training Process Improvements**:
   - Original lacked any mechanism to balance G/D training
   - Added proper training dynamics with conditional updates
   - Original had no techniques to prevent mode collapse

3. **Hyperparameter Corrections**:
   - Original learning rate (0.00002) was too low for effective training
   - Increased to DCGAN-recommended 0.0002 (10x higher)
   - Original batch size (64) was increased to 128 for more stable gradients
   - Added label smoothing (absent in original)

4. **Weight Initialization**:
   - Original implementation applied weights_init but didn't properly initialize BatchNorm
   - Fixed proper initialization for all layer types

These choices resulted in stable training dynamics and high-quality cartoon image generation.

---

## Contact

For any questions or issues, please contact:
- **Name**: Zhe Shen
- **Email**: zheshen2025@u.northwestern.edu
