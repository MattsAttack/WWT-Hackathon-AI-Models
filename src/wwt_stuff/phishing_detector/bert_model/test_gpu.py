import torch

has_cuda = torch.cuda.is_available()
print("CUDA is available: ", has_cuda)
# True


cuda_devices = torch.cuda.device_count() if has_cuda else 0
print("Number of CUDA devices: ", cuda_devices)

if has_cuda:
    print("CUDA current device: ", torch.cuda.current_device())
    # 0

    print("CUDA device name: ", torch.cuda.get_device_name(0))
    # 'NVIDIA GeForce RTX 3060'
