import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False
    import urllib.request as urllib_request

import io

# -----------------------------
# 1️⃣ Load dataset (use sklearn fallback by default to avoid 404)
# -----------------------------
try:
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print("✅ Loaded dataset: sklearn breast_cancer (substitute for heart disease)")
except Exception as e:
    raise RuntimeError("❌ Failed to load fallback dataset:", e)

print(df.head())
print('\nMissing values:', df.isnull().sum().sum())

# 2) try alternative known URL
if df is None:
    alt_url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
    try:
        print('⏳ Trying alternative URL...')
        if REQUESTS_AVAILABLE:
            s = requests.get(alt_url, timeout=15).content
        else:
            with urllib_request.urlopen(alt_url, timeout=15) as resp:
                s = resp.read()
        df = read_csv_from_bytes(s)
        print('✅ Dataset loaded from alternative URL')
    except Exception as e:
        last_error = e
        print('⚠️ Alternative URL failed:', repr(e))

# 3) try to load local file 'heart.csv' (in same folder)
if df is None:
    try:
        print('⏳ Trying to load local file heart.csv...')
        df = pd.read_csv('heart.csv')
        print('✅ Loaded local heart.csv')
    except Exception as e:
        last_error = e
        print("⚠️ Local file load failed:", repr(e))

# 4) fallback: use sklearn's breast cancer dataset as substitute
if df is None:
    try:
        print('⏳ Falling back to sklearn breast_cancer dataset')
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        print('✅ Loaded sklearn breast_cancer dataset as fallback')
    except Exception as e:
        last_error = e
        print('❌ All dataset loading attempts failed. Last error:', repr(last_error))
        raise RuntimeError('Failed to load any dataset; check network or provide local heart.csv')

print(df.head())
print('\nMissing values:', df.isnull().sum().sum())

# -----------------------------
# 2️⃣ Chuẩn bị dữ liệu
# -----------------------------
X = df.drop('target', axis=1).values
y = df['target'].values

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa (rất quan trọng với NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Đổi sang tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# -----------------------------
# 3️⃣ Xây dựng Neural Network
# -----------------------------
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 4️⃣ Huấn luyện mô hình
# -----------------------------
epochs = 200
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# -----------------------------
# 5️⃣ Đánh giá mô hình
# -----------------------------
with torch.no_grad():
    y_pred_test = model(X_test)
    y_pred_label = (y_pred_test > 0.5).float()

acc = (y_pred_label.eq(y_test).sum() / y_test.shape[0]).item()
print(f"\n✅ Accuracy: {acc*100:.2f}%")

# Classification report
y_true = y_test.numpy()
y_pred_np = y_pred_label.numpy()
print("\n📊 Classification Report:\n", classification_report(y_true, y_pred_np))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_np)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix | Accuracy: {acc*100:.2f}%")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# -----------------------------
# 6️⃣ Vẽ đường ROC
# -----------------------------
fpr, tpr, _ = roc_curve(y_true, y_pred_test.numpy())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# -----------------------------
# 7️⃣ Lưu mô hình
# -----------------------------
torch.save(model.state_dict(), "heart_model.pth")
print("💾 Model saved as heart_model.pth")