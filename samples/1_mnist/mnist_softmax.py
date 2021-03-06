# MNIST Softmax 
# Author: Django Peng
# Email: pengjingtian@huawei.com

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/tensorflow/mnist/data",
                    "Directory for storing mnist data")
FLAGS = flags.FLAGS

# Import data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# Create the model
# Image resolution is 28x28 = 784 pixels
x = tf.placeholder(tf.float32, [None, 784])
# Model variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
# True label 
y_ = tf.placeholder(tf.float32, [None, 10])
# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test 
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
