import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# 1️⃣ Thiết lập thiết bị
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")

# -----------------------------
# 2️⃣ Chuẩn bị dữ liệu
# -----------------------------
data_dir = "/Users/chuclinh/Downloads/2810/dogs-vs-cats/train"

transform = transforms.Compose([
    transforms.Resize((128, 128)),            # resize ảnh
    transforms.RandomHorizontalFlip(),        # lật ngẫu nhiên
    transforms.ToTensor(),                    # chuyển ảnh thành tensor
    transforms.Normalize((0.5, 0.5, 0.5),     # chuẩn hóa R,G,B
                         (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")

# -----------------------------
# 3️⃣ Định nghĩa mô hình CNN
# -----------------------------
class DogCatCNN(nn.Module):
    def __init__(self):
        super(DogCatCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DogCatCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# -----------------------------
# 4️⃣ Huấn luyện mô hình
# -----------------------------
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# 5️⃣ Đánh giá mô hình
# -----------------------------
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

# -----------------------------
# 6️⃣ Hiển thị kết quả dự đoán ảnh thật
# -----------------------------
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    img = images[i].cpu().numpy().transpose((1, 2, 0))
    img = (img * 0.5 + 0.5)  # unnormalize
    plt.imshow(img)
    title = f"Pred: {'Cat' if predicted[i]==0 else 'Dog'}"
    plt.title(title)
    plt.axis('off')
plt.show()

# -----------------------------
# 7️⃣ Lưu mô hình
# -----------------------------
torch.save(model.state_dict(), "dog_cat_cnn.pth")
print("💾 Model saved as dog_cat_cnn.pth")


# ============================
# 🔍 TEST ẢNH BẤT KỲ
# ============================
from PIL import Image

# Đường dẫn ảnh test
img_path = "/Users/chuclinh/Downloads/2810/dogs-vs-cats/train/dogs/dog.1.jpg"  # đổi sang ảnh bạn muốn test

# Load và biến đổi ảnh
img = Image.open(img_path)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
img_tensor = transform(img).unsqueeze(0).to(device)

# Dự đoán
model.eval()
with torch.no_grad():
    output = model(img_tensor)
    pred = torch.argmax(output, dim=1).item()

# In kết quả
classes = ['cat', 'dog']
print(f"👉 Prediction: {classes[pred].upper()}")