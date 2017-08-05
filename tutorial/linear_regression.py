import tensorflow as tf
import numpy as np


X = tf.placeholder(dtype=tf.float32)
W = tf.Variable([.0], dtype=tf.float32)
b = tf.Variable([.0], dtype=tf.float32)

linear_model = W * X + b
y = tf.placeholder(dtype=tf.float32)
loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Above graph can visualized as below
# (X, W, b) -> linear_model
#                 y
#              (linear_model, y) ->  loss
#                                    optimizer 
#                                    (loss, optimizer) -> train
# Now train is the single graph whose feed input is (X, y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1400):
        sess.run(train, {X: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {X: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print("weight: {}, bias: {}, loss: {}".format(curr_W, curr_b, curr_loss))
