import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Computer vision library
import data as data_module
import numpy as np

# Load the trained model
try:
    model = tf.keras.models.load_model("saved_model_svhn.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load an external image for prediction
# Example: place a digit image "digit.png" in the same folder
img_path = "digit.png"
if not os.path.exists(img_path):
    print(f"Error: Image file '{img_path}' not found!")
    exit(1)

try:
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Failed to load image")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (32, 32))  # Resize to 32x32 pixels
        image = image.astype('float32') / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        gray =cv2.cvtColor(image[0], cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        #find the contours of the digits
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits = []
        for cnt in contours[0]:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 5 and h >= 10:  # Filter out small contours
                digit = thresh[y:y+h, x:x+w]
                digit = cv2.resize(digit, (32, 32))
                digit = digit.astype('float32') / 255.0
                digit = np.expand_dims(digit, axis=-1)  # Add channel dimension
                digits.append((x, digit))  # Store x position for sorting
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

def augment_svhn_style(image):
    value = np.random.uniform(0.8, 1.2)
    image = np.clip(image * value, 0, 1)
    if np.random.rand() < 0.5:
        image = cv2.GaussianBlur(image, (3,3), 0)
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image + noise, 0, 1)
    return image



y_prob = model.predict(image)
pred_class = np.argmax(y_prob)
confidence = y_prob[0][pred_class]

# If individual digit contours were extracted, predict each one left-to-right.
if 'digits' in globals() and len(digits) > 0:
    digits_sorted = sorted(digits, key=lambda t: t[0])
    results = []
    for x, digit_img in digits_sorted:
        img = digit_img.copy()
        # Ensure 3 channels expected by the model
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)
        probs = model.predict(img)
        cls = int(np.argmax(probs[0]))
        conf = float(probs[0][cls])
        results.append((cls, conf))
    sequence = ''.join(str(r[0]) for r in results)
    confs = ', '.join(f"{r[1]:.2%}" for r in results)
    print(f"Predicted digit sequence: {sequence}")
    print(f"Confidences per digit: {confs}")
else:
    # Fallback to the whole-image prediction already computed above
    print(f"Predicted digit: {int(pred_class)} (confidence: {confidence:.2%})")