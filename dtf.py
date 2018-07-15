# import numpy as np
# import tensorflow as tf

# x = tf.placeholder(tf.float32, 100)

# mean = tf.reduce_mean(x)


# with tf.Session() as sess:
#     result = sess.run(mean, feed_dict={x: np.random.random(100)})
#     print(result)

import tensorflow as tf


cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

x = tf.constant(2)


with tf.device("/job:local/task:1"):
    y2 = x - 66

with tf.device("/job:local/task:0"):
    y1 = x + 300
    y = y1 + y2


with tf.Session("grpc://localhost:2222") as sess:
    result = sess.run(y)
    print(result)
    