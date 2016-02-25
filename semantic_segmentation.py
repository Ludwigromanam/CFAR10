import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from time import time

new_image_size = 72

# Model variables
image_size = 24
num_labels = 10
num_channels = 3
batch_size = 1
patch_size = 3
convdepth1 = 64
convdepth2 = 128
convdepth3 = 256

fcdepth = 1024


def inference(images):
  # Variables.
  l1conv1_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, convdepth1],
                                                   stddev=0.03))
  l1conv1_bias = tf.Variable(tf.zeros([convdepth1]))
  l1conv2_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth1, convdepth1],
                                                   stddev=0.03))
  l1conv2_bias = tf.Variable(tf.constant(0.01, shape=[convdepth1]))

  l2conv1_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth1, convdepth2],
                                                   stddev=0.03))
  l2conv1_bias = tf.Variable(tf.constant(0.01, shape=[convdepth2]))
  l2conv2_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth2, convdepth2],
                                                   stddev=0.03))
  l2conv2_bias = tf.Variable(tf.constant(0.01, shape=[convdepth2]))

  l3conv1_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth2, convdepth3],
                                                   stddev=0.03))
  l3conv1_bias = tf.Variable(tf.constant(0.01, shape=[convdepth3]))
  l3conv2_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth3, convdepth3],
                                                   stddev=0.03))
  l3conv2_bias = tf.Variable(tf.constant(0.01, shape=[convdepth3]))
  l3conv3_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth3, convdepth3],
                                                   stddev=0.03))
  l3conv3_bias = tf.Variable(tf.constant(0.01, shape=[convdepth3]))
  l3conv4_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth3, convdepth3],
                                                   stddev=0.03))
  l3conv4_bias = tf.Variable(tf.constant(0.01, shape=[convdepth3]))

  fc1_weight = tf.Variable(tf.truncated_normal([image_size/(2*2*2), image_size/(2*2*2), convdepth3, fcdepth],
                                               stddev=0.05))
  fc1_bias = tf.Variable(tf.constant(0.01, shape=[1, 1, fcdepth]))
  fc2_weight = tf.Variable(tf.truncated_normal([1, 1, fcdepth, fcdepth], stddev=0.05))
  fc2_bias = tf.Variable(tf.constant(0.01, shape=[1, 1, fcdepth]))
  out_weight = tf.Variable(tf.truncated_normal([1, 1, fcdepth, num_labels], stddev=0.05))
  out_bias = tf.Variable(tf.constant(0.0, shape=[1, 1, num_labels]))

  l1conv1 = tf.nn.conv2d(images, l1conv1_weight, [1, 1, 1, 1], padding='SAME')
  l1relu1 = tf.nn.relu(l1conv1 + l1conv1_bias)
  l1conv2 = tf.nn.conv2d(l1relu1, l1conv2_weight, [1, 1, 1, 1], padding='SAME')
  l1relu2 = tf.nn.relu(l1conv2 + l1conv2_bias)
  l1pool = tf.nn.max_pool(l1relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  l2conv1 = tf.nn.conv2d(l1pool, l2conv1_weight, [1, 1, 1, 1], padding='SAME')
  l2relu1 = tf.nn.relu(l2conv1 + l2conv1_bias)
  l2conv2 = tf.nn.conv2d(l2relu1, l2conv2_weight, [1, 1, 1, 1], padding='SAME')
  l2relu2 = tf.nn.relu(l2conv2 + l2conv2_bias)
  l2pool = tf.nn.max_pool(l2relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  l3conv1 = tf.nn.conv2d(l2pool, l3conv1_weight, [1, 1, 1, 1], padding='SAME')
  l3relu1 = tf.nn.relu(l3conv1 + l3conv1_bias)
  l3conv2 = tf.nn.conv2d(l3relu1, l3conv2_weight, [1, 1, 1, 1], padding='SAME')
  l3relu2 = tf.nn.relu(l3conv2 + l3conv2_bias)
  l3conv3 = tf.nn.conv2d(l3relu2, l3conv3_weight, [1, 1, 1, 1], padding='SAME')
  l3relu3 = tf.nn.relu(l3conv3 + l3conv3_bias)
  l3conv4 = tf.nn.conv2d(l3relu3, l3conv4_weight, [1, 1, 1, 1], padding='SAME')
  l3relu4 = tf.nn.relu(l3conv4 + l3conv4_bias)
  l3pool = tf.nn.max_pool(l3relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  fcconv1 = tf.nn.conv2d(l3pool, fc1_weight, [1, 1, 1, 1], padding='SAME')
  fcrelu1 = tf.nn.relu(fcconv1 + fc1_bias)
  fcconv2 = tf.nn.conv2d(fcrelu1, fc2_weight, [1, 1, 1, 1], padding='SAME')
  fcrelu2 = tf.nn.relu(fcconv2 + fc2_bias)

  output = tf.nn.conv2d(fcrelu2, out_weight, [1, 1, 1, 1], padding='SAME') + out_bias
  scaled_up_preds = tf.image.resize_images(output, new_image_size, new_image_size)
  logits = tf.reshape(scaled_up_preds, [-1, num_labels])

  return logits


def run_model_image(image):
    with tf.Graph().as_default():

      image = tf.reshape(image, [new_image_size, new_image_size, 3])
      image = tf.image.per_image_whitening(image)
      image = tf.reshape(image, [1, new_image_size, new_image_size, 3])
      image = tf.cast(image, tf.float32)

      train_prediction = inference(images=image)

      saver = tf.train.Saver(tf.all_variables())
      sess = tf.Session()
      saver.restore(sess=sess, save_path='vd_tmodel.ckpt')
      predictions = sess.run(train_prediction)


      return predictions


image = cv2.imread('/Users/pspitler3/Documents/caffe_images/dogandcat.jpg')
# image = image[40:320, :, :]
image = cv2.resize(image, (new_image_size, new_image_size), interpolation=cv2.INTER_AREA)
imdata = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

t = datetime.fromtimestamp(time(), None)

in_imdata = imdata.reshape((1, new_image_size, new_image_size, 3))

heatmap = run_model_image(image=in_imdata)

heatmap = heatmap.reshape(new_image_size, new_image_size, num_labels)
print np.min(heatmap)
print np.max(heatmap)

print datetime.fromtimestamp(time(), None) - t

fig, ax = plt.subplots(3, 4)
ax[0, 0].imshow(imdata)
ax[0, 0].set_title('Image')
ax[0, 1].imshow(heatmap[:, :, 0], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[0, 1].set_title('Airplane')
ax[0, 2].imshow(heatmap[:, :, 1], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[0, 2].set_title('Car')
ax[0, 3].imshow(heatmap[:, :, 2], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[0, 3].set_title('Bird')
ax[1, 0].imshow(heatmap[:, :, 3], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[1, 0].set_title('Cat')
ax[1, 1].imshow(heatmap[:, :, 4], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[1, 1].set_title('Deer')
ax[1, 2].imshow(heatmap[:, :, 5], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[1, 2].set_title('Dog')
ax[1, 3].imshow(heatmap[:, :, 6], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[1, 3].set_title('Frog')
ax[2, 0].imshow(heatmap[:, :, 7], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[2, 0].set_title('Horse')
ax[2, 1].imshow(heatmap[:, :, 8], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[2, 1].set_title('Boat')
ax[2, 2].imshow(heatmap[:, :, 9], vmin=np.min(heatmap), vmax=np.max(heatmap))
ax[2, 2].set_title('Truck')
plt.show()
