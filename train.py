import torch
import torch.nn as nn
from pathlib import Path

from model import SARModel
from data import build_dataloader
from utils import save_model

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Current CUDA device: {torch.cuda.current_device()}")


base_dir = Path("ROIs2017_winter_s2")
filepaths = list(base_dir.glob("*/*.tif"))

train_images = filepaths[:int(len(filepaths) * 0.1)]

train_dataloader = build_dataloader(train_images, shuffle=True)

print(len(train_images))

class g_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SARModel(device=device)

print(f"Generator device: {next(model.generator.parameters()).device}")
print(f"Discriminator device: {next(model.discriminator.parameters()).device}")

model.train(train_dataloader, epochs=64)
save_model(model.generator, "generator")

