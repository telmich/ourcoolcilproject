#!/usr/bin/env python

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow! It works!')
sess = tf.Session()
print(sess.run(hello))
