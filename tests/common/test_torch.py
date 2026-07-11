import torch

# 1. Check if an NVIDIA GPU with CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# 2. Dynamically assign the device
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# 3. Get information about your specific GPU
if cuda_available:
    print(f"Current Device ID: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 4. Move data or a model to the chosen device
x = torch.tensor([1.0, 2.0, 3.0])  # Created on CPU
x_gpu = x.to(device)  # Sent to GPU (if available)
print(f"Tensor device location: {x_gpu.device}")
