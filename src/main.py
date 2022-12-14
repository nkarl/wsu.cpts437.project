import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import os
import glob
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# data_path = os.path.join(os.getcwd(), 'data.adolescent.dep/')
# print(data_path)
# files = glob.glob(data_path + '*.nii.gz', recursive=False)
# # print(files)

# sample_file = files[0]
# import SimpleITK as sitk
# sitk_sample = sitk.ReadImage(sample_file)
# nii_img = sitk.GetArrayFromImage(sitk_sample)
# test = nii_img[:,:,30]
# print(nii_img.shape)
# plt.imshow(test)
# plt.show()

from process_scans import *
from aug import *

path_unmeddep = os.path.join(os.getcwd(), 'data.unmeddep/age/')
print(path_unmeddep)
scans_unmeddep = glob.glob(path_unmeddep + '*.nii.gz', recursive=False)
print(len(scans_unmeddep))
scans_unmeddep = np.array([process_scan(path) for path in scans_unmeddep])
labels_unmeddep = np.array([1 for _ in range(len(scans_unmeddep))])

path_other = os.path.join(os.getcwd(), 'data.other/age/')
print(path_other)
scans_other = glob.glob(path_other + '*.nii.gz', recursive=False)
print(len(scans_other))
scans_other = np.array([process_scan(path) for path in scans_other])
labels_other = np.array([1 for _ in range(len(scans_other))])

tt_split = int((len(scans_unmeddep) + len(scans_other)) * 0.7)
print(tt_split)

x_train = np.concatenate((scans_unmeddep[:tt_split], scans_other[:tt_split]), axis=0)
y_train = np.concatenate((labels_unmeddep[:tt_split], labels_other[:tt_split]), axis=0)
x_val = np.concatenate((scans_unmeddep[tt_split:], scans_other[tt_split:]), axis=0)
y_val = np.concatenate((labels_unmeddep[tt_split:], labels_other[tt_split:]), axis=0)

print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

def get_model(width=64, height=64, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()

initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
