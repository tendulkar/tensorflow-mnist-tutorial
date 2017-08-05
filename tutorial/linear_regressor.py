import tensorflow as tf
import numpy as np

# Features column names
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# estimator
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Data for training and evaluation
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.0])

input_fun = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000)
eval_input_fun = tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000)


x, y = input_fun()
print("x: {}, y: {}".format(x, y))
estimator.fit(input_fn=input_fun, steps=1000)

train_loss = estimator.evaluate(input_fn=input_fun)
eval_loss = estimator.evaluate(input_fn=eval_input_fun)

print("train loss: {}".format(train_loss))
print("eval loss: {}".format(eval_loss))


