# Fibonacci sequence
# Author: Django Peng
# Email: pengjingtian@huawei.com

import tensorflow as tf

# Iteration steps
steps = 20

# Define fibonacci array
fib = [1, 1]
# Define recursion formula
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = x + y

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in xrange(2,steps):
    f_i = sess.run(z, feed_dict={x: fib[i-2], y:fib[i-1]})
    fib.append(f_i)

print(fib)
