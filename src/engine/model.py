import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from engine.augment import *

def get_model(width=64, height=64, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


def build_model(learn_rate=0.0001, decay_steps=100000, decay_rate=0.96):
    """
    Build model.
    """
    model = get_model(width=64, height=64, depth=64)
    model.summary()
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        learn_rate,
        decay_steps,
        decay_rate,
        staircase=True
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )
    return model


# # Only rescale.
def dataset_loader(x, y, batch_size=2):
    """
    Define data loaders.
    """
    train_loader = tf.data.Dataset.from_tensor_slices((x, y))

    train_set = (
    train_loader.shuffle(len(x))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
    )

    return train_set


def train_model(model, train_dataset, validation_dataset, epoch=10):
    """
    Train the model given train and validation sets in some epochs.
    """
    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_image_classification.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    epochs = 10
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )
