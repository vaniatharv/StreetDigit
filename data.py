import tensorflow as tf
import tensorflow_datasets as tfds


(ds_train, ds_valid), ds_info = tfds.load(
    'svhn_cropped',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

x_train = []
y_train = []
x_valid = []
y_valid = []

for image, label in ds_train:
    x_train.append(image)
    y_train.append(label)

for image, label in ds_valid:
    x_valid.append(image)
    y_valid.append(label)

x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
x_valid = tf.convert_to_tensor(x_valid)
y_valid = tf.convert_to_tensor(y_valid)

import numpy as np
x_train = np.array(x_train)
y_train = np.array(y_train) 
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")