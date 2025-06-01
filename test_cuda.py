import torch

# Check if CUDA is available
print(f"CUDA is available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Create a test tensor on GPU
    x = torch.rand(5, 3)
    print("\nTest tensor on CPU:")
    print(x)
    
    # Move tensor to GPU
    x = x.cuda()
    print("\nTest tensor moved to GPU:")
    print(x)
else:
    print("CUDA is not available. Please check your PyTorch installation and CUDA setup.")
