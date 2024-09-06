import torch

def print_gpu_utilization():
    print(f"GPU utilization: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100:.2f}%")
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def get_device_info():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = f"Device used: {device}\n"
    if device.type == 'cuda':
        info += f"GPU model: {torch.cuda.get_device_name(0)}\n"
        info += f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n"
        info += f"CUDA version: {torch.version.cuda}"
    return info
