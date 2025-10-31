# 🐶 Dogs vs Cats Classification (PyTorch CNN)

A simple but solid CNN to classify images of dogs vs cats using PyTorch. Includes data prep, training, evaluation, and single-image prediction.

---

## 📦 What's inside

```
.
├─ day10_dog_cat_cnn.py   # Main training & inference script
├─ day9_transfer_learning.py # Transfer learning with ResNet18
├─ split_data.py          # Split raw Kaggle folder into train/val/test
├─ requirements.txt       # Dependencies (NumPy pinned <2)
├─ LICENSE                # MIT license
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
- See `requirements.txt` (NumPy is pinned `<2` for compatibility with some stacks)

Install:

```bash
pip install -r requirements.txt
```

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

## 🚀 Train (Day 10 CNN)

Set `data_dir` in `day10_dog_cat_cnn.py` (e.g., `data/train`) and run:

```bash
python day10_dog_cat_cnn.py
```

---

## 🚀 Train (Day 9 Transfer Learning)

`day9_transfer_learning.py` uses ResNet18. By default it tries pretrained weights; if download fails it trains from scratch and unfreezes the backbone automatically.

- Ensure dataset:
  - train: `data/train/{cats,dogs}`
  - val: `data/val/{cats,dogs}`

Run:

```bash
python day9_transfer_learning.py
```

Notes:
- For Apple Silicon (MPS) or CUDA, device is auto-selected.
- Transforms avoid NumPy dependency due to some envs; `requirements.txt` pins NumPy `<2` to be safe.

---

## 🔎 Single-image prediction (Day 10)

At the bottom of `day10_dog_cat_cnn.py`, set `img_path` to your image and run the script.

---

## 📝 License

MIT — see `LICENSE`.
