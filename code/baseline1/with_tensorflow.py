#!/usr/bin/env python3

import os

import tensorflow as tf

import util

DEFAULT_DATA_PATH = "../../cosmology_aux_data_170429"


    # x    = tf.placeholder(tf.float32, [None, util.DOWNSAMPLED_SIZE])
    # y_in = tf.placeholder(tf.float32, [None, 1])
    # W    = tf.Variable(tf.zeros([util.DOWNSAMPLED_SIZE, 1]))
    # b    = tf.Variable([0])  # TODO(kamila) if this is very different from 0, be surprised.
    # y    = tf.matmul(x, W) + b
    # loss = tf.reduce_mean(y - y_in)

    # train = optimizer.minimize(loss)


def stupid_linear(train_x, train_y, test_x, test_y):
    """Model that tries to predict the similarity score as a linear combination of pixel values.

    All pixel values.

    If this works, I'll eat my socks.
    """
    train_input_fn = tf.contrib.learn.io.numpy_input_fn({'x': train_x}, train_y, batch_size=100, num_epochs=1000)
    test_input_fn = tf.contrib.learn.io.numpy_input_fn({'x': test_x},  test_y,  batch_size=100)

    features = [tf.contrib.layers.real_valued_column('x', dimension=util.DOWNSAMPLED_SIZE)]
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
    estimator.fit(input_fn=train_input_fn, steps=1000)

    predictions = estimator.predict(input_fn=test_input_fn)

    print(estimator.evaluate(input_fn=test_input_fn))

    err, cnt = 0, 0
    for (y, y_in) in zip(predictions, test_y):
        print(y, y_in)
        err += abs(y - y_in)
        cnt += 1
    print('mean error', err/cnt)

def stupid_linear_exp():
    ...

if __name__ == "__main__":
    data = os.environ.get("COSMOLOGY_DATA", DEFAULT_DATA_PATH).strip()
    stupid_linear(*util.load_downsampled_train_test_images(data, 'scored'))
