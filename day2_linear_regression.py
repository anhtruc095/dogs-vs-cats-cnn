import torch
import matplotlib.pyplot as plt

# 1️⃣ Tạo dữ liệu giả lập (x, y)
# y = 2x + 1 + noise
torch.manual_seed(42)
x = torch.rand(100, 1) * 10  # giá trị x từ 0 -> 10
y = 2 * x + 1 + torch.randn(100, 1)  # thêm chút noise

# 2️⃣ Khởi tạo trọng số w và b (random)
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 3️⃣ Học mô hình
learning_rate = 0.01
epochs = 200

for epoch in range(epochs):
    # Dự đoán y_hat
    y_pred = w * x + b

    # Tính loss = MSE (mean squared error)
    loss = torch.mean((y_pred - y) ** 2)

    # Tính gradient
    loss.backward()

    # Cập nhật w, b thủ công
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Reset gradient về 0 để vòng sau tính lại
    w.grad.zero_()
    b.grad.zero_()

    # In loss mỗi 20 epoch
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

# 4️⃣ Kết quả cuối
print("\nFinal model parameters:")
print("w =", w.item())
print("b =", b.item())

# 5️⃣ Vẽ kết quả
with torch.no_grad():
    y_pred = w * x + b
plt.scatter(x, y, label="Data", color="blue")
plt.plot(x, y_pred, label="Fitted line", color="red")
plt.legend()
plt.show()

