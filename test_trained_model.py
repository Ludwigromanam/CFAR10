import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from PIL import Image

image_size = 24
num_labels = 10
num_channels = 3 # grayscale
batch_size = 1
patch_size = 5
depth = 64
layer1 = 384
layer2 = 192

graph = tf.Graph()
with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))

  # Variables.
  w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
  b1 = tf.Variable(tf.zeros([depth]))
  w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  b2 = tf.Variable(tf.constant(1.0, shape=[depth]))
  w3 = tf.Variable(tf.truncated_normal([image_size/(2*2), image_size/(2*2), depth, layer1], stddev=0.1))
  b3 = tf.Variable(tf.constant(1.0, shape=[layer1]))
  w4 = tf.Variable(tf.truncated_normal([1, 1, layer1, layer2], stddev=0.1))
  b4 = tf.Variable(tf.constant(1.0, shape=[layer2]))
  w5 = tf.Variable(tf.truncated_normal([1, 1, layer2, num_labels], stddev=0.1))
  b5 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  def test_model(data):
    conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv + b1)
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    conv = tf.nn.conv2d(norm, w2, [1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv + b2)
    norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(tf.nn.conv2d(pool, w3, [1, 1, 1, 1], padding='VALID') + b3)
    hidden2 = tf.nn.relu(tf.nn.conv2d(hidden, w4, [1, 1, 1, 1], padding='VALID') + b4)
    output = tf.nn.conv2d(hidden2, w5, [1, 1, 1, 1], padding='VALID') + b5
    return tf.reshape(output, [-1, num_labels])

  train_prediction = test_model(tf_train_dataset)


def run_model_image(graph, image):
    with tf.Session(graph=graph) as session:
        tf.train.Saver().restore(session, './model.ckpt')
        feed_dict = {tf_train_dataset: image}
        predictions = session.run([train_prediction], feed_dict=feed_dict)
        print predictions


image = cv2.imread('/Users/pspitler3/Documents/caffe_images/car.jpg')
#image = image[:, 60:420, :]
image = cv2.resize(image, (24, 24), interpolation=cv2.INTER_AREA)
cv2.imwrite('/Users/pspitler3/Documents/caffe_images/car_thumb.jpg', image)

imdata = mpimg.imread('/Users/pspitler3/Documents/caffe_images/car_thumb.jpg')

imdata = imdata.reshape((1, 24, 24, 3))

run_model_image(graph=graph, image=imdata)


