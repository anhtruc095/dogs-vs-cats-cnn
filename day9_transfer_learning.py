"""
Day 9: Transfer Learning with Pretrained ResNet18
Goal: Use a pretrained model (ResNet18) for a new classification task (e.g. cats vs dogs).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time

# ====== Setup ======
# Support MPS on Apple Silicon, else CUDA, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ====== Data transforms ======
# Force PIL-only tensorization to avoid NumPy usage in your torchvision version
class ToTensorNoNumpy(object):
    def __call__(self, img):
        # Ensure 3-channel RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        c = len(img.getbands())
        # Build tensor from raw bytes without NumPy
        byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        tensor = byte_tensor.view(h, w, c).permute(2, 0, 1).to(torch.float32) / 255.0
        return tensor

tensorize = ToTensorNoNumpy()

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    tensorize,
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    tensorize,
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Dataset & Dataloader ======
train_dir = "data/train"   # change path if needed
val_dir = "data/val"

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

# Use workers and pin_memory when on GPU/MPS for better throughput
num_workers = 0  # fallback to single-process loading to avoid NumPy issues
pin_memory = device.type in {"cuda", "mps"}
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# ====== Load Pretrained Model ======
# Use new weights API; fall back to no-weights (offline) if download fails
used_pretrained = True
try:
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
except Exception:
    used_pretrained = False
    try:
        # Newer API without weights (no download)
        from torchvision.models import resnet18
        model = resnet18(weights=None)
    except Exception:
        # Older API
        model = models.resnet18(pretrained=False)

# Freeze backbone only when using pretrained weights
if used_pretrained:
    for p in model.parameters():
        p.requires_grad = False

# Replace last layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Always train the classifier head
for p in model.fc.parameters():
    p.requires_grad = True

model = model.to(device)

# ====== Loss & Optimizer ======
criterion = nn.CrossEntropyLoss()
# If not pretrained, train all params; else just final layer
params_to_optimize = model.parameters() if not used_pretrained else model.fc.parameters()
optimizer = optim.Adam(params_to_optimize, lr=0.001)

# ====== Training Loop ======
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        # Train
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / max(1, len(train_loader.dataset))

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = (correct / total) if total > 0 else 0.0

        print(f"Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_resnet18.pth")
            print("✅ Saved best model")

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    since = time.time()
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)
    print(f"\nTotal time: {(time.time() - since)/60:.1f} min")