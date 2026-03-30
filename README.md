# StreetDigit (SVHN Digit Recognition)

StreetDigit is a deep learning project that trains a Convolutional Neural Network (CNN) to recognize digits from the **Street View House Numbers (SVHN)** dataset (`svhn_cropped`).  
It includes simple scripts for **data loading**, **model training**, and **predicting digits from an external image**.

---

## Project Structure

- `data.py` — downloads/loads the SVHN dataset using `tensorflow_datasets` and prepares `x_train`, `y_train`, `x_valid`, `y_valid`.
- `model.py` — builds and trains a CNN classifier and saves the trained model to disk.
- `prediction.py` — loads the saved model and predicts digit(s) from an input image file (default: `digit.png`).
- `README.md` — documentation.

---

## Requirements

Install dependencies (recommended in a virtual environment):

```bash
pip install tensorflow tensorflow-datasets numpy opencv-python matplotlib
```

Notes:
- `prediction.py` uses **OpenCV** (`cv2`) for image processing.
- The SVHN dataset is fetched automatically via `tensorflow_datasets`.

---

## 1) Load the Dataset

`data.py` uses TensorFlow Datasets to load:

- Dataset: `svhn_cropped`
- Splits used:
  - `train` → training set
  - `test` → used as validation set in this repo

To verify the dataset loads correctly:

```bash
python data.py
```

It will print shapes like:

- `x_train shape: (...)`
- `x_valid shape: (...)`

---

## 2) Train the Model

Run training from `model.py`:

```bash
python model.py
```

What it does:
- Imports `data.py` to access `x_train/y_train` and `x_valid/y_valid`
- Normalizes images to `[0, 1]` (if needed)
- Builds a small CNN:
  - Conv2D(32) → MaxPool
  - Conv2D(64) → MaxPool
  - Dense(128) → Dense(num_classes, softmax)
- Trains for `epochs=10` (default in `__main__`)
- Saves the trained model to:

```text
saved_model_svhn.h5
```

If you want to train with a different number of epochs or change save name, edit the bottom of `model.py`.

---

## 3) Predict Digits From an Image

### Prepare an image
Place an image named `digit.png` in the repository root directory (same folder as `prediction.py`).

Then run:

```bash
python prediction.py
```

Behavior:
- Loads the trained model from `saved_model_svhn.h5`
- Loads `digit.png`, resizes it to **32×32**, normalizes it, and predicts a digit
- Also attempts to **extract multiple digit contours** (threshold + contours) and predict them left-to-right as a digit sequence

Example output:
- `Predicted digit: 5 (confidence: 97.12%)`
or
- `Predicted digit sequence: 120`
- `Confidences per digit: 91.23%, 88.05%, 94.11%`

---

## Notes / Limitations

- SVHN images are RGB (32×32×3). The model expects **3 channels**.
- The multi-digit contour logic in `prediction.py` is a best-effort heuristic; results depend heavily on the quality/background of `digit.png`.
- This repo currently uses a simple CNN and SGD optimizer; accuracy can be improved with deeper architectures, augmentation, Adam optimizer, and callbacks.

---

## License

Add a license if you plan to publish or reuse this project (e.g., MIT, Apache-2.0).
