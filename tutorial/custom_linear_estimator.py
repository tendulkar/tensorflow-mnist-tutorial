import tensorflow as tf
import numpy as np


session = tf.InteractiveSession()


def model(features, labels, mode):
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b

    loss = tf.reduce_sum(tf.square(y - labels))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    global_step = tf.train.get_global_step()
    print("W: {}, x: {}, b: {}, y: {}, loss: {}, global step: {}".format(W, features['x'], b, y, loss, global_step))

    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)


# Prepare data generators
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fun = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, 3, num_epochs=1000)
eval_fun = tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval, 3, num_epochs=1000)

# train the model
estimator = tf.contrib.learn.Estimator(model_fn=model)
estimator.fit(input_fn=input_fun, steps=1000)

# train loss, eval loss
train_loss = estimator.evaluate(input_fn=input_fun)
print("Training is done")
eval_loss = estimator.evaluate(input_fn=eval_fun)

print("training loss: {}, eval loss: {}".format(train_loss, eval_loss))

session.close()
