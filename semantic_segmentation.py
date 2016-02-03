import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from PIL import Image

image_size = 32
new_image_size = 64
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
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, new_image_size, new_image_size, num_channels))

  # Variables.
  w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
  print w1.get_shape().as_list()
  b1 = tf.Variable(tf.zeros([depth]))
  w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  print w2.get_shape().as_list()
  b2 = tf.Variable(tf.constant(1.0, shape=[depth]))
  w3 = tf.Variable(tf.truncated_normal([image_size / (2 * 2) * image_size / (2 * 2) * depth, layer1], stddev=0.1))
  print w3.get_shape().as_list()
  b3 = tf.Variable(tf.constant(1.0, shape=[layer1]))
  w4 = tf.Variable(tf.truncated_normal([layer1, layer2], stddev=0.1))
  print w4.get_shape().as_list()
  b4 = tf.Variable(tf.constant(1.0, shape=[layer2]))
  w5 = tf.Variable(tf.truncated_normal([layer2, num_labels], stddev=0.1))
  print w5.get_shape().as_list()
  b5 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  def test_model(data):
    conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    print conv.get_shape().as_list()
    relu = tf.nn.relu(conv + b1)
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print pool.get_shape().as_list()
    conv = tf.nn.conv2d(pool, w2, [1, 1, 1, 1], padding='SAME')
    print conv.get_shape().as_list()
    relu = tf.nn.relu(conv + b2)
    pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print pool.get_shape().as_list()
    conv = tf.nn.conv2d(pool, tf.reshape(w3, (image_size/(2*2), image_size/(2*2), depth, layer1)),
                        [1, 1, 1, 1], padding='VALID')
    print conv.get_shape().as_list()
    relu = tf.nn.relu(conv + tf.reshape(b3, (1, 1, layer1)))
    conv = tf.nn.conv2d(relu, tf.reshape(w4, (1, 1, layer1, layer2)),
                        [1, 1, 1, 1], padding='VALID')
    print conv.get_shape().as_list()
    relu = tf.nn.relu(conv + tf.reshape(b4, (1, 1, layer2)))
    output = tf.nn.conv2d(relu, tf.reshape(w5, (1, 1, layer2, num_labels)),
                          [1, 1, 1, 1], padding='VALID')
    print output.get_shape().as_list()
    return output + tf.reshape(b5, (1, 1, num_labels))

    # shape = pool.get_shape().as_list()
    # reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
    # hidden = tf.nn.relu(tf.matmul(reshape, w3) + b3)
    # hidden2 = tf.nn.relu(tf.matmul(hidden, w4) + b4)
    # return tf.matmul(hidden2, w5) + b5

  train_prediction = test_model(tf_train_dataset)


def run_model_image(graph, image):
    with tf.Session(graph=graph) as session:
        tf.train.Saver().restore(session, './model.ckpt')
        feed_dict = {tf_train_dataset : image}
        predictions = session.run([train_prediction], feed_dict=feed_dict)
        return np.array(predictions)


image = cv2.imread('/Users/pspitler3/Documents/caffe_images/dogsandcar.jpg')
image = image[:, 60:60+360, :]
image = cv2.resize(image, (new_image_size, new_image_size), interpolation=cv2.INTER_AREA)
cv2.imwrite('/Users/pspitler3/Documents/caffe_images/dogandcar_thumb.jpg', image)

imdata = mpimg.imread('/Users/pspitler3/Documents/caffe_images/dogandcar_thumb.jpg')

imdata = imdata.reshape((1, new_image_size, new_image_size, 3))

heatmap = run_model_image(graph=graph, image=imdata)

heatmap = np.reshape(heatmap, (9, 9, 10))
print np.shape(heatmap)

plt.imshow(heatmap[:, :, 0]), plt.show()
plt.imshow(heatmap[:, :, 1]), plt.show()
plt.imshow(heatmap[:, :, 2]), plt.show()
plt.imshow(heatmap[:, :, 3]), plt.show()
plt.imshow(heatmap[:, :, 4]), plt.show()
plt.imshow(heatmap[:, :, 5]), plt.show()
plt.imshow(heatmap[:, :, 6]), plt.show()
plt.imshow(heatmap[:, :, 7]), plt.show()
plt.imshow(heatmap[:, :, 8]), plt.show()
plt.imshow(heatmap[:, :, 9]), plt.show()