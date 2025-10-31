import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Tạo dữ liệu (data)
# -----------------------------
# Tạo 100 điểm ngẫu nhiên x
x = torch.randn(100, 1)

# Tạo nhãn y: nếu x > 0 thì 1, ngược lại 0 (nhị phân)
y = (x > 0).float()

# -----------------------------
# 2️⃣ Xây dựng mô hình Logistic Regression
# -----------------------------
# Mô hình: y_pred = sigmoid(w*x + b)
model = nn.Sequential(
    nn.Linear(1, 1),   # Lớp tuyến tính
    nn.Sigmoid()       # Hàm kích hoạt sigmoid
)

# -----------------------------
# 3️⃣ Khai báo hàm mất mát và bộ tối ưu
# -----------------------------
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Gradient Descent

# -----------------------------
# 4️⃣ Huấn luyện mô hình
# -----------------------------
for epoch in range(200):
    # Forward: dự đoán
    y_pred = model(x)

    # Tính loss
    loss = criterion(y_pred, y)

    # Backward: cập nhật trọng số
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # In ra mỗi 20 epoch
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# -----------------------------
# 5️⃣ In kết quả cuối cùng
# -----------------------------
print("\nFinal model parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# -----------------------------
# 6️⃣ Trực quan hóa dữ liệu
# -----------------------------
plt.scatter(x.numpy(), y.numpy(), label='Actual data')
plt.scatter(x.numpy(), model(x).detach().numpy(), label='Predicted', color='red')
plt.legend()
plt.title("Logistic Regression Result")
plt.show()