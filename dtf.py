import tensorflow as tf

tf.app.flags.DEFINE_string('job_name', '', 'One of local worker')
tf.app.flags.DEFINE_string('local', '', """Comma-separated list of hostname:port for the """)

tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of local/replica running the training')
tf.app.flags.DEFINE_integer('constant_id', 0, 'the constant we want to run')

FLAGS = tf.app.flags.FLAGS

local_host = FLAGS.local.split(',')

cluster = tf.train.ClusterSpec({"local": local_host})
server = tf.train.Server(cluster,job_name=FLAGS.job_name, task_index=FLAGS.task_id, protocol='grpc+gdr') # default protocol is 'grpc'

# server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

with tf.Session(server.target) as sess:
    if(FLAGS.constant_id == 0):
        with tf.device('/job:local/task:'+str(FLAGS.task_id)):
            const1 = tf.constant("Hello I am the first constant")
            print sess.run(const1)
    if (FLAGS.constant_id == 1):
        with tf.device('/job:local/task:'+str(FLAGS.task_id)):
            const2 = tf.constant("Hello I am the second constant")
            print sess.run(const2)