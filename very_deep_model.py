import numpy as np
import tensorflow as tf
import time
import os
import argparse
from read_data import distorted_inputs, inputs

#Global variables
num_labels = 10
valid_records = 5000
test_records = 10000
train_records = 45000

# Model variables
image_size = 24
num_labels = 10
num_channels = 3 # grayscale
batch_size = 100
patch_size = 3
convdepth1 = 64
convdepth2 = 128
convdepth3 = 256

fcdepth = 1024

conv_dprob = 0.75
hidden_dprob = 0.5

#lr = 0.000001
#num_epochs = 350


def accuracy(predictions, labels):
  labels = tf.cast(labels, tf.int32)
  matches = tf.nn.in_top_k(predictions=predictions, targets=tf.arg_max(labels, 1), k=1)
  return matches


def evaluate(test_set, path):
    with tf.Graph().as_default():

      images, labels = inputs(test_set)

      logits = inference(train=False, images=images)
      test_acc = accuracy(logits, labels)

      saver = tf.train.Saver(tf.all_variables())

      sess = tf.Session()
      coord = tf.train.Coordinator()
      saver.restore(sess=sess, save_path=path)

      threads = tf.train.start_queue_runners(coord=coord, sess=sess)

      try:
        true_count = 0
        if test_set == 'valid.tfrecords':
          num_records = valid_records
        else:
          num_records = test_records

        step = 0
        while step < int(num_records/batch_size):
          acc = sess.run(test_acc)
          true_count += np.sum(acc)
          step += 1

      except tf.errors.OutOfRangeError as e:
        print 'Issues: ', e
      finally:
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        sess.close()

      return 100 * (float(true_count)/num_records)


def inference(train, images):

  l1conv1_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, convdepth1],
                                                   stddev=2.0/(24*24*3 + 24*24*64)))
  l1conv1_bias = tf.Variable(tf.zeros([convdepth1]))
  l1conv2_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth1, convdepth1],
                                                   stddev=2.0/(24*24*64 + 24*24*64)))
  l1conv2_bias = tf.Variable(tf.constant(0.001, shape=[convdepth1]))

  l2conv1_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth1, convdepth2],
                                                   stddev=2.0/(12*12*64 + 12*12*128)))
  l2conv1_bias = tf.Variable(tf.constant(0.001, shape=[convdepth2]))
  l2conv2_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth2, convdepth2],
                                                   stddev=2.0/(12*12*128 + 12*12*128)))
  l2conv2_bias = tf.Variable(tf.constant(0.001, shape=[convdepth2]))

  l3conv1_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth2, convdepth3],
                                                   stddev=2.0/(6*6*128 + 6*6*256)))
  l3conv1_bias = tf.Variable(tf.constant(0.001, shape=[convdepth3]))
  l3conv2_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth3, convdepth3],
                                                   stddev=2.0/(6*6*256 + 6*6*256)))
  l3conv2_bias = tf.Variable(tf.constant(0.001, shape=[convdepth3]))
  l3conv3_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth3, convdepth3],
                                                   stddev=2.0/(6*6*256 + 6*6*256)))
  l3conv3_bias = tf.Variable(tf.constant(0.001, shape=[convdepth3]))
  l3conv4_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, convdepth3, convdepth3],
                                                   stddev=2.0/(6*6*256 + 6*6*256)))
  l3conv4_bias = tf.Variable(tf.constant(0.001, shape=[convdepth3]))

  fc1_weight = tf.Variable(tf.truncated_normal([image_size/(2*2*2), image_size/(2*2*2), convdepth3, fcdepth],
                                               stddev=2.0/(6*6*256 + fcdepth)))
  fc1_bias = tf.Variable(tf.constant(0.001, shape=[1, 1, fcdepth]))
  fc2_weight = tf.Variable(tf.truncated_normal([1, 1, fcdepth, fcdepth], stddev=2.0/(fcdepth + fcdepth)))
  fc2_bias = tf.Variable(tf.constant(0.001, shape=[1, 1, fcdepth]))
  out_weight = tf.Variable(tf.truncated_normal([1, 1, fcdepth, num_labels], stddev=2.0/(fcdepth + num_labels)))
  out_bias = tf.Variable(tf.constant(0.0, shape=[1, 1, num_labels]))

  # Model.
  def train_model(data):
    l1conv1 = tf.nn.conv2d(data, l1conv1_weight, [1, 1, 1, 1], padding='SAME')
    l1relu1 = tf.nn.relu(l1conv1 + l1conv1_bias)
    l1conv2 = tf.nn.conv2d(l1relu1, l1conv2_weight, [1, 1, 1, 1], padding='SAME')
    l1relu2 = tf.nn.relu(l1conv2 + l1conv2_bias)
    l1pool = tf.nn.max_pool(l1relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1dropout = tf.nn.dropout(l1pool, conv_dprob)

    l2conv1 = tf.nn.conv2d(l1dropout, l2conv1_weight, [1, 1, 1, 1], padding='SAME')
    l2relu1 = tf.nn.relu(l2conv1 + l2conv1_bias)
    l2conv2 = tf.nn.conv2d(l2relu1, l2conv2_weight, [1, 1, 1, 1], padding='SAME')
    l2relu2 = tf.nn.relu(l2conv2 + l2conv2_bias)
    l2pool = tf.nn.max_pool(l2relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2dropout = tf.nn.dropout(l2pool, conv_dprob)

    l3conv1 = tf.nn.conv2d(l2dropout, l3conv1_weight, [1, 1, 1, 1], padding='SAME')
    l3relu1 = tf.nn.relu(l3conv1 + l3conv1_bias)
    l3conv2 = tf.nn.conv2d(l3relu1, l3conv2_weight, [1, 1, 1, 1], padding='SAME')
    l3relu2 = tf.nn.relu(l3conv2 + l3conv2_bias)
    l3conv3 = tf.nn.conv2d(l3relu2, l3conv3_weight, [1, 1, 1, 1], padding='SAME')
    l3relu3 = tf.nn.relu(l3conv3 + l3conv3_bias)
    l3conv4 = tf.nn.conv2d(l3relu3, l3conv4_weight, [1, 1, 1, 1], padding='SAME')
    l3relu4 = tf.nn.relu(l3conv4 + l3conv4_bias)
    l3pool = tf.nn.max_pool(l3relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3dropout = tf.nn.dropout(l3pool, conv_dprob)

    fcconv1 = tf.nn.conv2d(l3dropout, fc1_weight, [1, 1, 1, 1], padding='VALID')
    fcrelu1 = tf.nn.relu(fcconv1 + fc1_bias)
    fcdrop1 = tf.nn.dropout(fcrelu1, hidden_dprob)
    fcconv2 = tf.nn.conv2d(fcdrop1, fc2_weight, [1, 1, 1, 1], padding='VALID')
    fcrelu2 = tf.nn.relu(fcconv2 + fc2_bias)
    fcdrop2 = tf.nn.dropout(fcrelu2, hidden_dprob)

    output = tf.nn.conv2d(fcdrop2, out_weight, [1, 1, 1, 1], padding='VALID')
    output = tf.reshape(output + out_bias, [-1, num_labels])
    return output

  def test_model(data):
    l1conv1 = tf.nn.conv2d(data, l1conv1_weight, [1, 1, 1, 1], padding='SAME')
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

    fcconv1 = tf.nn.conv2d(l3pool, fc1_weight, [1, 1, 1, 1], padding='VALID')
    fcrelu1 = tf.nn.relu(fcconv1 + fc1_bias)
    fcconv2 = tf.nn.conv2d(fcrelu1, fc2_weight, [1, 1, 1, 1], padding='VALID')
    fcrelu2 = tf.nn.relu(fcconv2 + fc2_bias)

    output = tf.nn.conv2d(fcrelu2, out_weight, [1, 1, 1, 1], padding='VALID')
    output = tf.reshape(output + out_bias, [-1, num_labels])
    return output

  if train:
    logits = train_model(images)
  else:
    logits = test_model(images)

  return logits


def calc_loss(logits, labels):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
  return loss


def training(loss, learning_rate):
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  return optimizer


def run_training(path):
  with tf.Graph().as_default():

    train_images, train_labels = distorted_inputs(num_threads=10)

    logits = inference(train=True, images=train_images)
    loss = calc_loss(logits, train_labels)
    train_op = training(loss, learning_rate=lr)

    saver = tf.train.Saver(tf.all_variables())
    init_op = tf.initialize_all_variables()

    sess = tf.Session()

    if os.path.isfile(path):
      saver.restore(sess=sess, save_path=path)
      print 'Model Restored'
    else:
      sess.run(init_op)
      print 'Model Initialized'

    tf.train.start_queue_runners(sess=sess)

    for step in xrange(int((num_epochs * train_records)/batch_size)):

      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      if step % 225 == 0 or step == int((num_epochs * train_records)/batch_size):
        print "------------------------------------------"
        print "Examples/sec: ", batch_size/duration
        print "Sec/batch: ", float(duration)
        print "Current epoch: ", (float(step) * batch_size) / train_records
        print "Current learning rate: ", lr
        print "Minibatch loss at step", step, ":", loss_value
      if step % 900 == 0 or step == int((num_epochs * train_records)/batch_size) - 1:
        save_path = saver.save(sess, path)
        print "Model saved in file: ", save_path
        print "Validation accuracy: ", evaluate('valid.tfrecords', path)

    print "===================================="
    print "Test accuracy: ", evaluate('test.tfrecords', path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_file', help='The checkpoint file to write output to.')
  parser.add_argument('num_epochs', help='The number of epochs to run for.')
  parser.add_argument('lr', help='The learning rate to run on.')
  args = parser.parse_args()
  num_epochs = int(args.num_epochs)
  lr = float(args.lr)
  run_training(args.checkpoint_file)