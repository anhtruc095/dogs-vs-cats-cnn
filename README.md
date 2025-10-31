# 🐶 Dogs vs Cats Classification (PyTorch CNN)

A simple but solid CNN to classify images of dogs vs cats using PyTorch. Includes data prep, training, evaluation, and single-image prediction.

---

## 📦 What's inside

```
.
├─ day10_dog_cat_cnn.py   # Main training & inference script
├─ split_data.py          # Split raw Kaggle folder into train/val/test
├─ README.md              # This file
└─ data/                  # Created by split_data.py (ignored by Git)
   ├─ train/
   │  ├─ cats/
   │  └─ dogs/
   ├─ val/
   │  ├─ cats/
   │  └─ dogs/
   └─ test/
      ├─ cats/
      └─ dogs/
```

Note: Large datasets, zip files, and model artifacts are ignored via `.gitignore`.

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch + Torchvision
- Pillow, NumPy, Matplotlib

Install quickly:

```bash
pip install torch torchvision pillow numpy matplotlib
```

Or use your preferred environment/manager (conda, uv, pipx, etc.).

---

## 📥 Dataset

Use the Kaggle “Dogs vs Cats” dataset. You can place the original Kaggle train folder here:

```
dogs-vs-cats/train   # contains cat.0.jpg, dog.0.jpg, ... (flat folder)
```

Then run the splitter to create ImageFolder-compatible directories:

```bash
python split_data.py
```

This creates `data/train|val|test` with `cats/` and `dogs/` subfolders.

---

## 🚀 Train

By default, `day10_dog_cat_cnn.py` expects a training folder with class subfolders. Recommended:

- Set the `data_dir` to the training folder you want to use, for example:
  - `data_dir = "data/train"` (after running `split_data.py`), or
  - point it to another ImageFolder-compatible path.

Start training:

```bash
python day10_dog_cat_cnn.py
```

- Uses GPU if available, otherwise CPU.
- Trains a small 3-layer CNN for a few epochs.
- Saves weights to `dog_cat_cnn.pth`.

---

## ✅ Evaluate

The script automatically evaluates on the held-out split and prints Test Accuracy. It also visualizes a few predictions from the test loader.

---

## 🔎 Single-image prediction

At the bottom of `day10_dog_cat_cnn.py`, set `img_path` to any JPG and run the script. Example:

```python
img_path = "data/test/dogs/dog.1.jpg"  # change to your image
```

The script will load the image, run inference, and print the predicted label.

---

## 📝 Notes

- If you used the raw Kaggle flat folder, always run `split_data.py` first to build class subfolders.
- `.gitignore` excludes `data/`, `*.zip`, and `*.pth` to avoid pushing large artifacts.
- If you see very low accuracy, ensure your `data_dir` points to the folder with `cats/` and `dogs/` subfolders and that transforms are applied correctly.

---

## 🔗 Repository

https://github.com/anhtruc095/dogs-vs-cats-cnn
