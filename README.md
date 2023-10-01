# Convolutional Auto-Encoder for Image Reconstruction and Denoising

This project involves building and training a convolutional auto-encoder model for the tasks of image reconstruction and denoising using the CIFAR10 dataset.

## Table of Contents
- [Image Reconstruction](#image-reconstruction)
- [Denoising](#denoising)
- [Results](#results)
- [Libraries Used](#libraries-used)
- [References](#references)

## Image Reconstruction

### Objective
Design and train a convolutional auto-encoder for image reconstruction on the CIFAR10 dataset.

### Methodology
- Utilized the CIFAR10 data loader from PyTorch.
- The auto-encoder model consists of 2 encoder blocks (2D convolution) and 2 decoder blocks (transpose 2D convolution).
- Channel configurations: 
    - Encoder: (3,8), (8,8)
    - Decoder: (8,8), (8,3)
- Activation functions: Relu and Tanh.
- Loss: Mean Squared Error (MSE).
- Optimizer: Adam.

### Results
- The model converged at a loss of `0.005` by epoch 9.
<p align="center">
  <img src="https://github.com/arushi2509/Autoencoder-for-Image-Reconstruction-and-Image-Denoising/assets/69112495/c4c5fdae-563c-41c8-bb98-61515f56cebe">
</p>
- Sample reconstructions:
  ![Original vs Reconstructed Images](path/to/your/figure2.png)

## Denoising

### Objective
Train the convolutional auto-encoder for image denoising on CIFAR10.

### Methodology
- The input images were perturbed with Gaussian noise (mean=0, variance=0.1) using the `torch.randn` function.
- Same model and training configurations as the image reconstruction task.
  
### Results
- The model converged at a loss of `0.017` by epoch 8.
- ![Convergence of Loss for Denoising](path/to/your/figure3.png)
- Sample denoising results:
  ![Original vs Noisy vs Denoised Images](path/to/your/figure4.png)
- Performance Metrics:
    - Average PSNR: `23.860`
    - Average SSIM: `0.867`

## Libraries Used
- PyTorch
- skimage (for PSNR and SSIM calculations)

## References
1. [PyTorch CIFAR10 DataLoader](https://pytorch.org/vision/stable/datasets.html#cifar)
