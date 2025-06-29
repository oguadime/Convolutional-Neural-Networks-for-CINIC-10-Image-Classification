from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from einops.layers.torch import Rearrange
import pandas as pd
import matplotlib.pyplot as plt

# 1) Data & Transforms
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.47889522, 0.47227842, 0.43047404],
        std=[0.24205776, 0.23828046, 0.25874835],
    )
])

cinic10_dir = "./data/cinic10/"

train_data = ImageFolder(root=Path(cinic10_dir) / "train", transform=transform)
valid_data = ImageFolder(root=Path(cinic10_dir) / "valid", transform=transform)
test_data  = ImageFolder(root=Path(cinic10_dir) / "test",  transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=64)
test_loader  = DataLoader(test_data,  batch_size=64)

# 2) Parameterized CNN
class CINICClassifier(nn.Module):
    def __init__(self, channels, num_classes=10, kernel_size=3):
        """
        channels: list of ints, e.g. [32,64] or [64,128,256]
        """
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # final spatial dims after pooling
        final_spatial = 32 // (2 ** len(channels))
        hidden_dim = in_ch * final_spatial * final_spatial

        self.fc = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

# 3) Training & Validation
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn   = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        total_loss += loss.item() * bs
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += bs

    return total_loss/total, 100*correct/total

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        bs = inputs.size(0)
        total_loss += loss.item() * bs
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += bs

    return total_loss/total, 100*correct/total

# 4) Define search space
channel_configs = [
    [32, 64],
    [64, 128],
    [32, 64, 128],
    [64, 128, 256],
    [32, 64, 128, 256],
    [64, 128, 256, 512],
]

results = []
epochs = 5  # or whatever you like

# 5) Sweep
for channels in channel_configs:
    print(f"\n=== Testing channels={channels} ===")
    model = CINICClassifier(channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train/val
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer)
        va_loss, va_acc = validate(model, valid_loader)
        print(f"Epoch {epoch:>2} | "
              f"Train: {tr_loss:.3f}, {tr_acc:5.1f}% | "
              f" Val: {va_loss:.3f}, {va_acc:5.1f}%")

    # test
    te_loss, te_acc = validate(model, test_loader)
    print(f"→ Test accuracy: {te_acc:5.1f}%\n")

    results.append({
        "depth": len(channels),
        "channels": "-".join(map(str, channels)),
        "test_acc": te_acc,
    })

# 6) Summarize
df = pd.DataFrame(results).sort_values("test_acc", ascending=False)
print("All results:\n", df, sep="")
best = df.iloc[0]
print(f"\nBest config: depth={best.depth}, channels=[{best.channels}] → {best.test_acc:.1f}%")

# 7) (Optional) Plot the best run's curves
# You could store histories per config and plot similarly to the original.
