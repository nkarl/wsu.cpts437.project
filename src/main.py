import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from engine.model import build_model

from engine.process_scans import *
from engine.model import *


# Load the data files.
target='Age' if len(sys.argv) < 2 else sys.argv[1]
scans_unmeddep, labels_unmeddep, scans_other, labels_other = datasets(target=target)
tt_split = int((len(scans_unmeddep) + len(scans_other)) * 0.7)
x_train = np.concatenate((scans_unmeddep[:tt_split], scans_other[:tt_split]), axis=0)
y_train = np.concatenate((labels_unmeddep[:tt_split], labels_other[:tt_split]), axis=0)
x_val = np.concatenate((scans_unmeddep[tt_split:], scans_other[tt_split:]), axis=0)
y_val = np.concatenate((labels_unmeddep[tt_split:], labels_other[tt_split:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)


# Load the training set and validating set.
train_dataset = dataset_loader(x_train, y_train)
validation_dataset = dataset_loader(x_val, y_val)
#Build the model
class ImageSize:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        pass


dim=ImageSize(64, 64, 64)

model = build_model(dim, learn_rate=0.0001, decay_steps=100000, decay_rate=0.96)
# Train the model, doing validation at the end of each epoch
train_model(model, train_dataset, validation_dataset, epoch=20)


# Visualize the model performance
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()
for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])


# Load best weights.
model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]
class_names = ["unmeddep", "other"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )
