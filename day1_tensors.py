import torch
import numpy as np

# Tạo tensor từ list Python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print("x_data:\n", x_data)

# Tạo tensor ngẫu nhiên (random)
x_random = torch.rand((2, 3))
print("\nRandom tensor:\n", x_random)

# Tạo tensor toàn 0 hoặc toàn 1
x_ones = torch.ones((2, 2))
x_zeros = torch.zeros((2, 2))
print("\nOnes:\n", x_ones)
print("\nZeros:\n", x_zeros)

# Tạo tensor từ numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("\nTensor from numpy:\n", x_np)

print("Shape:", x_data.shape)
print("Data type:", x_data.dtype)
print("Device:", x_data.device)

x = torch.rand(2, 2)
y = torch.rand(2, 2)

print("\nx:", x)
print("y:", y)

# Cộng
print("\nAdd:", x + y)
print("Add (hàm):", torch.add(x, y))

# Nhân phần tử
print("\nMultiply:", x * y)

# Ma trận nhân
print("\nMatrix multiply:", torch.mm(x, y))

# Lấy giá trị max
print("\nMax value:", x.max())

x = torch.arange(1, 10)  # tensor từ 1 → 9
print("\nOriginal:", x)

x_reshaped = x.reshape(3, 3)
print("Reshaped (3x3):\n", x_reshaped)

# Thêm hoặc xóa chiều
x_unsqueeze = x.unsqueeze(0)  # thêm 1 chiều ở đầu
print("\nUnsqueeze (thêm 1 chiều):", x_unsqueeze.shape)

x_squeeze = x_unsqueeze.squeeze(0)  # bỏ chiều
print("Squeeze (bỏ chiều):", x_squeeze.shape)

# Tensor sang numpy
x = torch.ones(3, 3)
x_np = x.numpy()
print("\nTensor to NumPy:\n", x_np)

# NumPy sang tensor
np_arr = np.zeros((2, 2))
x_from_np = torch.from_numpy(np_arr)
print("\nNumPy to Tensor:\n", x_from_np)