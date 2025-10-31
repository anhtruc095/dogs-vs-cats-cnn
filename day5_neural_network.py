import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Tạo dữ liệu phi tuyến
# -----------------------------
# x là 1D từ -1 đến 1
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# y là hàm phi tuyến: y = x^2 + noise
y = x.pow(2) + 0.2 * torch.rand(x.size())

# -----------------------------
# 2️⃣ Xây dựng mạng neural (MLP)
# -----------------------------
# Mạng có 2 hidden layers:
# input (1) → hidden1 (10 neurons, ReLU) → hidden2 (5 neurons, ReLU) → output (1)
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# -----------------------------
# 3️⃣ Định nghĩa loss và optimizer
# -----------------------------
criterion = nn.MSELoss()  # Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# -----------------------------
# 4️⃣ Train model
# -----------------------------
for epoch in range(500):
    # Forward
    y_pred = model(x)

    # Loss
    loss = criterion(y_pred, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # In ra mỗi 50 epoch
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# -----------------------------
# 5️⃣ Kết quả
# -----------------------------
print("\nFinal model parameters:")
for name, param in model.named_parameters():
    print(name, param.data.shape)

# -----------------------------
# 6️⃣ Trực quan hóa kết quả
# -----------------------------
plt.scatter(x.numpy(), y.numpy(), label='Real data')
plt.plot(x.numpy(), model(x).detach().numpy(), color='red', label='Model prediction')
plt.legend()
plt.title("Neural Network Fit for Non-linear Data")
plt.show()