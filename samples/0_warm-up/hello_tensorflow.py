import tensorflow as tf

# Define a variable of string dtype
str = tf.Variable("Hello TensorFlow")

# Define a session
sess = tf.Session()

# Initialize all variables by session run
sess.run(tf.initialize_all_variables())

sess.run(str)
