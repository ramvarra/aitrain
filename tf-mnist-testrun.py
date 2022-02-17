import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime

import util

EPOCHS = 5
BATCH_SIZE = 64
VERBOSE = 1
NUM_CLASSES = 10 # number of digits 0..9
N_HIDDEN = 128
VALIDATION_SPLIT = .2 # portion of train data reserved for validation
DROP_OUT = 0.3

def show_stats(name, v):
    print(f"{name}.shape: {v.shape} dtype: {v.dtype}, min: {v.min()}, max: {v.max()}, mean: {v.mean()}")


util.show_tf_info(tf)

# load data
print("Loading training data")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# reshape the X into 2D, convert to float32, normalize by dividing by 255
x_train, x_test = (x.reshape(x.shape[0], -1).astype(np.float32)/255 for x in (x_train, x_test))
show_stats("x_train", x_train)
show_stats("x_test", x_test)

# One hot encode Y
y_train, y_test = (tf.keras.utils.to_categorical(y, NUM_CLASSES) for y in (y_train, y_test))
show_stats("y_train", y_train)
show_stats("y_test", y_test)

# Build the model
model = tf.keras.models.Sequential(name='MNIST-2H')
for layer in [
    tf.keras.layers.Dense(N_HIDDEN, name='hidden_layer_01', input_shape=(x_train.shape[1],),  activation='relu'),
    tf.keras.layers.Dropout(DROP_OUT),
    tf.keras.layers.Dense(N_HIDDEN, name='hidden_layer_02', activation='relu'),
    tf.keras.layers.Dropout(DROP_OUT),
    tf.keras.layers.Dense(N_HIDDEN, name='hidden_layer_03', activation='relu'),
    tf.keras.layers.Dropout(DROP_OUT),
    tf.keras.layers.Dense(NUM_CLASSES, name='ouput_layer', activation='softmax'),
]: model.add(layer)

# compile the model
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

print("Starting the training")
ts_start = datetime.now()
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
elapsed = datetime.now() - ts_start
print(f"Tooks {elapsed} to train {EPOCHS} epocs. Time/epoch = {elapsed/EPOCHS} seconds")

print("Evaluating the mode:")
loss, acc = model.evaluate(x_test, y_test)
print(f"Model Accuracy = {acc}")
