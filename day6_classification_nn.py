import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# -----------------------------
# 1️⃣ Tạo dữ liệu 2D (phi tuyến)
# -----------------------------
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # chuyển thành cột

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 2️⃣ Xây dựng Neural Network
# -----------------------------
# Input: 2 features → Hidden(10) → Hidden(5) → Output(1)
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()  # đầu ra 0–1 cho phân loại nhị phân
)

# -----------------------------
# 3️⃣ Định nghĩa loss & optimizer
# -----------------------------
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 4️⃣ Train model
# -----------------------------
for epoch in range(1000):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/1000 | Loss: {loss.item():.4f}")

# -----------------------------
# 5️⃣ Đánh giá mô hình
# -----------------------------
with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred_label = (y_test_pred > 0.5).float()
    acc = (y_test_pred_label.eq(y_test).sum() / y_test.shape[0]).item()
    print(f"\n✅ Accuracy on test data: {acc*100:.2f}%")

# -----------------------------
# 6️⃣ Vẽ vùng quyết định (Decision Boundary)
# -----------------------------
import numpy as np

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

with torch.no_grad():
    Z = model(grid)
    Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.squeeze(), cmap=plt.cm.coolwarm, edgecolors='k')
plt.title(f"Decision Boundary | Accuracy: {acc*100:.2f}%")
plt.show()