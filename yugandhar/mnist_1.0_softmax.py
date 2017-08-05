import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
# Initialize variables and place holders
# A place holder is nothing but a node, which can take input as argument specified by variable name
# where as variable (again a node) changes frequently and can be updated partially (a part of it)
# Following linear model is a graph

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# computes predicted values using softmax regression
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]),  W) + b)

# cross entropy is loss
cross_entropy = - tf.reduce_sum(Y_ * tf.log(Y))

is_correct = tf.equal(tf.arg_max(Y, 1), tf.arg_max(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

# Create session which encapsulates control and context of graph
# And initialize variables which can be updated when graph performs computation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, Y_: batch_Y}
        sess.run(train_step, feed_dict=train_data)

        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print("Train accuracy: {}, cross entropy: {}".format(a, c))
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print("Test accuracy: {}, cross entropy: {}".format(a, c))
