import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# -----------------------------
# 1️⃣ Tạo dữ liệu
# -----------------------------
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# 2️⃣ Mô hình Neural Network
# -----------------------------
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 3️⃣ Huấn luyện
# -----------------------------
for epoch in range(1000):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/1000 | Loss: {loss.item():.4f}")

# -----------------------------
# 4️⃣ Đánh giá mô hình
# -----------------------------
with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred_label = (y_test_pred > 0.5).float()
    acc = (y_test_pred_label.eq(y_test).sum() / y_test.shape[0]).item()
    print(f"\n✅ Accuracy: {acc*100:.2f}%")

# Chuyển sang numpy để dùng sklearn metrics
y_true = y_test.numpy().astype(int)
y_pred_np = y_test_pred_label.numpy().astype(int)

# Confusion Matrix & Report
cm = confusion_matrix(y_true, y_pred_np)
print("\n📊 Classification Report:\n", classification_report(y_true, y_pred_np))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix | Accuracy: {acc*100:.2f}%")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -----------------------------
# 5️⃣ Lưu mô hình
# -----------------------------
torch.save(model.state_dict(), "moon_model.pth")
print("💾 Model saved as moon_model.pth")

# -----------------------------
# 6️⃣ Load lại mô hình để dự đoán
# -----------------------------
loaded_model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()
)
loaded_model.load_state_dict(torch.load("moon_model.pth"))
loaded_model.eval()
print("✅ Model loaded successfully!")

# Test thử dự đoán 1 điểm mới
new_point = torch.tensor([[1.5, 0.2]])
pred = loaded_model(new_point)
print(f"\n🔮 Dự đoán cho điểm {new_point.tolist()}: {float(pred):.3f}")
