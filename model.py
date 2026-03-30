import data as data_module
import tensorflow as tf
from tensorflow import keras
# expose submodules from the already-imported tensorflow.keras to avoid
# "Import 'tensorflow.keras' could not be resolved" in some linters/environments
layers = keras.layers
models = keras.models
optimizers = keras.optimizers
import numpy as np

def prepare_data(module):
    """Load arrays from the provided module and prepare them for training.

    Expects the module to provide: x_train, y_train, x_valid, y_valid.
    Returns: (x_train, y_train), (x_valid, y_valid)

    Labels are returned as integer class indices (suitable for
    'sparse_categorical_crossentropy' used in the training code).
    """

    # load arrays
    x_train = getattr(module, 'x_train')
    y_train = getattr(module, 'y_train')
    x_valid = getattr(module, 'x_valid')
    y_valid = getattr(module, 'y_valid')

    # ensure numeric arrays
    x_train = np.asarray(x_train)
    x_valid = np.asarray(x_valid)
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)

    # cast images to float32 and scale to [0, 1] if necessary
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    if x_train.max() > 1.0:
        x_train /= 255.0
    if x_valid.max() > 1.0:
        x_valid /= 255.0

    # ensure channel dimension is present: (H, W) -> (H, W, 1)
    if x_train.ndim == 3:
        x_train = x_train[..., None]
    if x_valid.ndim == 3:
        x_valid = x_valid[..., None]

    # ensure labels are integer class indices
    # if labels are one-hot, convert to indices
    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_valid.ndim > 1:
        y_valid = np.argmax(y_valid, axis=1)

    y_train = y_train.astype('int32')
    y_valid = y_valid.astype('int32')

    return (x_train, y_train), (x_valid, y_valid)

def build_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.SGD(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train(epochs, save_path=None):
    (x_train, y_train), (x_valid, y_valid) = prepare_data(data_module)
    # labels are integer class indices (1D). Derive num_classes safely.
    if y_train.size == 0:
        num_classes = 1
    else:
        num_classes = int(np.max(y_train)) + 1
    model = build_model(input_shape=x_train.shape[1:], num_classes=num_classes)
    model.summary()
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=32)

    if save_path:
        # save the whole model (architecture + weights + optimizer state)
        model.save(save_path)

    return model, history


if __name__ == '__main__':
    # TensorFlow will add the correct extension for the saved model format
    save_file = 'saved_model_svhn.h5'
    train(epochs=10, save_path=save_file)