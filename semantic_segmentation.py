import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

image_size = 24
new_image_size = 48
num_labels = 10
num_channels = 3
batch_size = 1
patch_size = 5
depth = 64
layer1 = 384
layer2 = 192


def inference(images):
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

  conv = tf.nn.conv2d(images, w1, [1, 1, 1, 1], padding='SAME')
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
      saver.restore(sess=sess, save_path='tmodel.ckpt')
      predictions = sess.run(train_prediction)


      return predictions


#image = cv2.imread('/Users/pspitler3/Documents/caffe_images/dogsandcar.jpg')
#image = image[30:330, 90:390, :]
#image = cv2.resize(image, (new_image_size, new_image_size), interpolation=cv2.INTER_AREA)
#cv2.imwrite('/Users/pspitler3/Documents/caffe_images/dogsandcar_thumb.jpg', image)

imdata = mpimg.imread('/Users/pspitler3/Documents/caffe_images/dogandcat_thumb.jpg')

# train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_cfar10_data()
# imdata = train_dataset[2, :, :, :]
# imdata = np.reshape(imdata, (image_size, image_size, 3))
# label = train_labels[2, :]

imdata = imdata.reshape((1, new_image_size, new_image_size, 3))
print np.shape(imdata)

heatmap = run_model_image(image=imdata)
print np.shape(heatmap)

heatmap = heatmap.reshape(new_image_size, new_image_size, num_labels)
print np.min(heatmap)
print np.max(heatmap)

plt.imshow(heatmap[:, :, 0], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()
plt.imshow(heatmap[:, :, 1], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()
plt.imshow(heatmap[:, :, 2], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()
plt.imshow(heatmap[:, :, 3], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()
plt.imshow(heatmap[:, :, 4], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()
plt.imshow(heatmap[:, :, 5], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()
plt.imshow(heatmap[:, :, 6], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()
plt.imshow(heatmap[:, :, 7], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()
plt.imshow(heatmap[:, :, 8], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()
plt.imshow(heatmap[:, :, 9], vmin=np.min(heatmap), vmax=np.max(heatmap)), plt.show()