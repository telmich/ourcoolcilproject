#!/usr/bin/env python3

import os

import tensorflow as tf

import util

DEFAULT_DATA_PATH = "../../cosmology_aux_data_170429"
N_INPUT_THREADS   = 4

    # x    = tf.placeholder(tf.float32, [None, util.DOWNSAMPLED_SIZE])
    # y_in = tf.placeholder(tf.float32, [None, 1])
    # W    = tf.Variable(tf.zeros([util.DOWNSAMPLED_SIZE, 1]))
    # b    = tf.Variable([0])  # TODO(kamila) if this is very different from 0, be surprised.
    # y    = tf.matmul(x, W) + b
    # loss = tf.reduce_mean(y - y_in)

    # train = optimizer.minimize(loss)

def get_data(sess, data_path, subset):
    def enqueue_op():
        utils.enqueue_images(data_path, subset, input_queue, flatten=True)
    queue_runner = tf.train.QueueRunner(input_queue, [enqueue_op] * N_INPUT_THREADS)
    coord = tf.train.Coordinator()
    enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)
    sess.run()

    # coord.join(enqueue_threads)



def try_some_estimators(sess, data_path):
    """Model that tries several ready-made estimators from tf.contrib.learn."""
    NSTEPS=1000
    input_queue  = tf.FIFOQueue(dtypes=(tf.string, tf.uint32), name='input queue [{}]'.format(subset))
    sess.run(get_data(sess, data_path, 'scored'))


    def test_queue_op():
        img_id, data, score = data_queue.dequeue()
        print(img_id, score)

    try:
        for step in range(NSTEPS):
            if coord.should_stop():
                break
            sess.run(test_queue_op)
    except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)

    # # train_input_fn, test_input_fn = util.images_stoch_input(data_path, 'scored')
    # # train_queue, test_queue = tf.RandomShuffleQueue(name='train queue'), tf.RandomShuffleQueue(name='test queue')
    # features = [tf.contrib.layers.real_valued_column('pixels', dimension=util.IMAGE_SIZE)]
    # estimators = [
    #     tf.contrib.learn.DNNRegressor(feature_columns=features, hidden_units=[256,128,32]),
    #     tf.contrib.learn.LinearRegressor(feature_columns=features),
    # ]

    # for estimator in estimators:
    #     estimator.fit(input_fn=train_input_fn, steps=NSTEPS)
    #     print(estimator.evaluate(input_fn=test_input_fn))
    #     predictions = estimator.predict(input_fn=test_input_fn)

    #     err, cnt = 0, 0
    #     for (y, y_in) in zip(predictions, test_y):
    #         print(y, y_in)
    #         err += abs(y - y_in)
    #         cnt += 1
    #     print('===== mean error =====', err/cnt)
    #     print()

if __name__ == "__main__":
    data = os.environ.get("COSMOLOGY_DATA", DEFAULT_DATA_PATH).strip()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    try_some_estimators(sess, data)
