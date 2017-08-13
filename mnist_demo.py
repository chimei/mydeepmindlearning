#!/usr/bin/python
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  if i % 20 == 0:
    sys.stdout.write('.')
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # if i<2:
  #   print("i = " + str(i) + "\nbatch_xs = " + str(batch_xs) + "\nbatch_ys =" + str(batch_ys) + "\n\n")
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print("")

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))