import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1️⃣ Tạo dữ liệu giả lập
torch.manual_seed(42)
x = torch.rand(100, 1) * 10
y = 2 * x + 1 + torch.randn(100, 1)

# 2️⃣ Xây mô hình Linear Regression
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 3️⃣ Chọn hàm mất mát (loss) và thuật toán tối ưu
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4️⃣ Training loop
epochs = 200
for epoch in range(epochs):
    # Forward pass
    y_pred = model(x)

    # Tính loss
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # In kết quả
    if (epoch + 1) % 20 == 0:
        w, b = model.linear.weight.item(), model.linear.bias.item()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {w:.4f}, b = {b:.4f}")

# 5️⃣ Kết quả cuối cùng
w, b = model.linear.weight.item(), model.linear.bias.item()
print("\nFinal parameters:")
print("w =", w)
print("b =", b)

# 6️⃣ Vẽ kết quả
with torch.no_grad():
    y_pred = model(x)
plt.scatter(x, y, label="Data", color="blue")
plt.plot(x, y_pred, label="Fitted line", color="red")
plt.legend()
plt.show()

