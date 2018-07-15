import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, 100)

mean = tf.reduce_mean(x)


with tf.Session() as sess:
    result = sess.run(mean, feed_dict={x: np.random.random(100)})
    print(result)