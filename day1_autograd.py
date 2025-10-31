import torch

# 1️⃣ Khởi tạo tensor và bật tính toán gradient
x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
print("x:", x)

# 2️⃣ Định nghĩa hàm y = x^2 + 3x
y = x ** 2 + 3 * x
print("y:", y)

# 3️⃣ Lấy tổng để có một giá trị scalar (bắt buộc để backward)
z = y.sum()
print("z (sum of y):", z)

# 4️⃣ Tính gradient dz/dx
z.backward()

# 5️⃣ In ra gradient
print("Gradient (dz/dx):", x.grad)

# 6️⃣ Ngắt theo dõi gradient (ví dụ trong quá trình inference)
with torch.no_grad():
    y2 = x ** 2 + 3 * x
    print("Tính toán không theo dõi gradient:", y2)

# 7️⃣ Kiểm tra lại thuộc tính requires_grad
print("x.requires_grad:", x.requires_grad)
print("y2.requires_grad:", y2.requires_grad)