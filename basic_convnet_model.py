import numpy as np
import tensorflow as tf
import time
import logging
from read_data import distorted_inputs, inputs
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

patch_size = 5
depth = 64
layer1 = 384
layer2 = 192
hidden_dprob = 0.7
IMAGE_SIZE = 24
num_labels = 10
batch_size = 100
num_channels = 3

tf.app.flags.DEFINE_integer('num_epochs', 350, 'The number of validations records')
FLAGS = tf.app.flags.FLAGS


def accuracy(predictions, labels):
  labels = tf.cast(labels, tf.int32)
  matches = tf.nn.in_top_k(predictions=predictions, targets=tf.arg_max(labels, 1), k=1)
  return matches


def evaluate(test_set):
    with tf.Graph().as_default():

      images, labels = inputs(test_set)

      logits = inference(train=False, images=images)
      test_acc = accuracy(logits, labels)

      saver = tf.train.Saver(tf.all_variables())

      sess = tf.Session()
      coord = tf.train.Coordinator()
      saver.restore(sess=sess, save_path='model.ckpt')

      threads = tf.train.start_queue_runners(coord=coord, sess=sess)

      try:
        true_count = 0
        if test_set == 'valid.tfrecords':
          num_records = FLAGS.valid_records
        else:
          num_records = FLAGS.test_records

        step = 0
        while step < int(num_records/FLAGS.batch_size):
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
  # Variables.
  w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
  b1 = tf.Variable(tf.zeros([depth]))
  w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  b2 = tf.Variable(tf.constant(1.0, shape=[depth]))
  w3 = tf.Variable(tf.truncated_normal([IMAGE_SIZE/(2*2), IMAGE_SIZE/(2*2), depth, layer1], stddev=0.1))
  b3 = tf.Variable(tf.constant(1.0, shape=[layer1]))
  w4 = tf.Variable(tf.truncated_normal([1, 1, layer1, layer2], stddev=0.1))
  b4 = tf.Variable(tf.constant(1.0, shape=[layer2]))
  w5 = tf.Variable(tf.truncated_normal([1, 1, layer2, num_labels], stddev=0.1))
  b5 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  # Model.
  def train_model(data):
    conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv + b1)
    relu_dropout = tf.nn.dropout(relu, hidden_dprob)
    pool = tf.nn.max_pool(relu_dropout, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    conv = tf.nn.conv2d(norm, w2, [1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv + b2)
    norm = tf.nn.lrn(relu, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    relu_dropout = tf.nn.dropout(norm, hidden_dprob)
    pool = tf.nn.max_pool(relu_dropout, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(tf.nn.conv2d(pool, w3, [1, 1, 1, 1], padding='VALID') + b3)
    hidden_dropout = tf.nn.dropout(hidden, hidden_dprob)
    hidden2 = tf.nn.relu(tf.nn.conv2d(hidden_dropout, w4, [1, 1, 1, 1], padding='VALID') + b4)
    hidden2_dropout = tf.nn.dropout(hidden2, hidden_dprob)
    output = tf.nn.conv2d(hidden2_dropout, w5, [1, 1, 1, 1], padding='VALID') + b5
    return tf.reshape(output, [-1, num_labels])

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

  if train:
    logits = train_model(images)
  else:
    logits = test_model(images)

  return logits


def calc_loss(logits, labels):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
  return loss


def training(loss, learning_rate):
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(learning_rate,
                                             global_step * batch_size,
                                             FLAGS.train_records * 20,
                                             0.99,
                                             staircase=True)
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  return optimizer, learning_rate


def run_training():
  with tf.Graph().as_default():

    train_images, train_labels = distorted_inputs(num_epochs=FLAGS.num_epochs, num_threads=25)

    logits = inference(train=True, images=train_images)
    loss = calc_loss(logits, train_labels)
    train_op, curr_lr = training(loss, learning_rate=0.02)

    saver = tf.train.Saver(tf.all_variables())

    init_op = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)

    for step in xrange(int((FLAGS.num_epochs * FLAGS.train_records)/FLAGS.batch_size)):

      start_time = time.time()
      _, lr, loss_value = sess.run([train_op, curr_lr, loss])
      duration = time.time() - start_time

      if step % 225 == 0 or step == int((FLAGS.num_epochs * FLAGS.train_records)/FLAGS.batch_size):
        print "------------------------------------------"
        print "Examples/sec: ", FLAGS.batch_size/duration
        print "Sec/batch: ", float(duration)
        print "Current epoch: ", (float(step) * batch_size) / FLAGS.train_records
        print "Current learning rate: ", lr
        print "Minibatch loss at step", step, ":", loss_value
      if step % 900 == 0 or step == int((FLAGS.num_epochs * FLAGS.train_records)/FLAGS.batch_size) - 1:
        save_path = saver.save(sess, "./model.ckpt")
        print "Model saved in file: ", save_path
        print "Validation accuracy: ", evaluate('valid.tfrecords')

    print "===================================="
    print "Validation accuracy: ", evaluate('valid.tfrecords')
    print "Test accuracy: ", evaluate('test.tfrecords')


def main():
  run_training()
