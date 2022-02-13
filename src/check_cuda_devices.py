"""
Very small script to quickly display cuda information according to torch.
This is useful to be sure of a cuda setup before running on multi-device machine.
"""

import os

import torch

print(f"ENV \"CUDA_VISIBLE_DEVICES\": {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"Num cuda devices: {torch.cuda.device_count()}")
print(f"Current cuda device: {torch.cuda.current_device()}")
