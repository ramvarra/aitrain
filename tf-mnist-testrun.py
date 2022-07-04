
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime

DEFAULT_EPOCHS = 50
BATCH_SIZE = 64
NUM_CLASSES = 10    # number of digits 0..9
N_HIDDEN = 128
VALIDATION_SPLIT = .2 # portion of train data reserved for validation
DROP_OUT = 0.3


def show_stats(name, v):
    print(f"{name}.shape: {v.shape} dtype: {v.dtype}, min: {v.min()}, max: {v.max()}, mean: {v.mean()}")

def show_tf_info(tf):
    '''
    Display information about the TF package
    '''
    print('TF Version:', tf.__version__)
    print('Logical Devices:', tf.config.list_logical_devices())
    print('Physical Devices:', tf.config.list_physical_devices())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow MNIST Training and Evaluation Tool')
    parser.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS, help=f'number of epochs - default {DEFAULT_EPOCHS}')
    parser.add_argument("--verbose", "-v", type=int, default=0, help=f'verbose - 0 (default): silent, 1: progress bar 2: detailed log')
    args = parser.parse_args()
    show_tf_info(tf)

    # load data
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
    model = tf.keras.models.Sequential(name='MNIST-3H')
    for layer in [
        tf.keras.layers.Dense(N_HIDDEN, name='hidden_relu_01', input_shape=(x_train.shape[1],),  activation='relu'),
        tf.keras.layers.Dropout(DROP_OUT),
        tf.keras.layers.Dense(N_HIDDEN, name='hidden_relu_02', activation='relu'),
        tf.keras.layers.Dropout(DROP_OUT),
        tf.keras.layers.Dense(N_HIDDEN, name='hidden_relu_03', activation='relu'),
        tf.keras.layers.Dropout(DROP_OUT),
        tf.keras.layers.Dense(NUM_CLASSES, name='output_softmax', activation='softmax'),
    ]: model.add(layer)

    # compile the model
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    print(f"Starting the training - {args.epochs} epochs")
    ts_start = datetime.now()
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=args.epochs, verbose=args.verbose, validation_split=VALIDATION_SPLIT)
    elapsed = datetime.now() - ts_start
    print(f"Training time {elapsed} for {args.epochs} epocs. Time/epoch = {elapsed/args.epochs} seconds")

    print("Evaluating the model")

    ts_start = datetime.now()
    loss, acc = model.evaluate(x_test, y_test, verbose=args.verbose)
    elapsed = datetime.now() - ts_start
    print(f"Evaluation time {elapsed} for {len(x_test)} samples.")
    print(f"Test Accuracy = {acc}")
