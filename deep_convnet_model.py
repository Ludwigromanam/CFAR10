import numpy as np
import tensorflow as tf
import time
import os
import argparse
from read_data import distorted_inputs, inputs

# Global variables
num_labels = 10
valid_records = 5000
test_records = 10000
train_records = 45000

# Model variables
image_size = 24
num_channels = 3
batch_size = 100
patch_size = 3
depth1 = 32
depth2 = 64
depth3 = 384
depth4 = 192

conv_dprob = 0.8
hidden_dprob = 0.7


# lr = 0.000001
# num_epochs = 350


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
            while step < int(num_records / batch_size):
                acc = sess.run(test_acc)
                true_count += np.sum(acc)
                step += 1

        except tf.errors.OutOfRangeError as e:
            print 'Issues: ', e
        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            sess.close()

        return 100 * (float(true_count) / num_records)


def inference(train, images):
    conv1_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth1], stddev=0.1))
    conv1_bias = tf.Variable(tf.zeros([depth1]))
    conv2_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth1, depth1], stddev=0.1))
    conv2_bias = tf.Variable(tf.constant(0.1, shape=[depth1]))

    conv3_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth1, depth2], stddev=0.1))
    conv3_bias = tf.Variable(tf.constant(0.1, shape=[depth2]))
    conv4_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth2, depth2], stddev=0.1))
    conv4_bias = tf.Variable(tf.constant(0.1, shape=[depth2]))

    conv5_weight = tf.Variable(
        tf.truncated_normal([image_size / (2 * 2), image_size / (2 * 2), depth2, depth3], stddev=0.1))
    conv5_bias = tf.Variable(tf.constant(0.1, shape=[1, 1, depth3]))
    conv6_weight = tf.Variable(tf.truncated_normal([1, 1, depth3, depth4], stddev=0.1))
    conv6_bias = tf.Variable(tf.constant(0.1, shape=[1, 1, depth4]))
    conv7_weight = tf.Variable(tf.truncated_normal([1, 1, depth4, num_labels], stddev=0.1))
    conv7_bias = tf.Variable(tf.constant(0.0, shape=[1, 1, num_labels]))

    # Model.
    def train_model(data):
        conv1 = tf.nn.conv2d(data, conv1_weight, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(conv1 + conv1_bias)
        relu1_dropout = tf.nn.dropout(relu1, conv_dprob)
        conv2 = tf.nn.conv2d(relu1_dropout, conv2_weight, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(conv2 + conv2_bias)
        relu2_dropout = tf.nn.dropout(relu2, conv_dprob)
        pool1 = tf.nn.max_pool(relu2_dropout, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3 = tf.nn.conv2d(pool1, conv3_weight, [1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(conv3 + conv3_bias)
        relu3_dropout = tf.nn.dropout(relu3, conv_dprob)
        conv4 = tf.nn.conv2d(relu3_dropout, conv4_weight, [1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(conv4 + conv4_bias)
        relu4_dropout = tf.nn.dropout(relu4, conv_dprob)
        pool2 = tf.nn.max_pool(relu4_dropout, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        hidden_conv1 = tf.nn.conv2d(pool2, conv5_weight, [1, 1, 1, 1], padding='VALID')
        hidden_relu1 = tf.nn.relu(hidden_conv1 + conv5_bias)
        hidden_relu1_dropout = tf.nn.dropout(hidden_relu1, hidden_dprob)
        hidden_conv2 = tf.nn.conv2d(hidden_relu1_dropout, conv6_weight, [1, 1, 1, 1], padding='VALID')
        hidden_relu2 = tf.nn.relu(hidden_conv2 + conv6_bias)
        hidden_relu2_dropout = tf.nn.dropout(hidden_relu2, hidden_dprob)

        output = tf.nn.conv2d(hidden_relu2_dropout, conv7_weight, [1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output + conv7_bias, [-1, num_labels])
        return output

    def test_model(data):
        conv1 = tf.nn.conv2d(data, conv1_weight, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(conv1 + conv1_bias)
        conv2 = tf.nn.conv2d(relu1, conv2_weight, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(conv2 + conv2_bias)
        pool1 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3 = tf.nn.conv2d(pool1, conv3_weight, [1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(conv3 + conv3_bias)
        conv4 = tf.nn.conv2d(relu3, conv4_weight, [1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(conv4 + conv4_bias)
        pool2 = tf.nn.max_pool(relu4, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

        hidden_conv1 = tf.nn.conv2d(pool2, conv5_weight, [1, 1, 1, 1], padding='VALID')
        hidden_relu1 = tf.nn.relu(hidden_conv1 + conv5_bias)
        hidden_conv2 = tf.nn.conv2d(hidden_relu1, conv6_weight, [1, 1, 1, 1], padding='VALID')
        hidden_relu2 = tf.nn.relu(hidden_conv2 + conv6_bias)

        output = tf.nn.conv2d(hidden_relu2, conv7_weight, [1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output + conv7_bias, [-1, num_labels])
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

        for step in xrange(int((num_epochs * train_records) / batch_size)):

            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            if step % 225 == 0 or step == int((num_epochs * train_records) / batch_size):
                print "------------------------------------------"
                print "Examples/sec: ", batch_size / duration
                print "Sec/batch: ", float(duration)
                print "Current epoch: ", (float(step) * batch_size) / train_records
                print "Current learning rate: ", lr
                print "Minibatch loss at step", step, ":", loss_value
            if step % 900 == 0 or step == int((num_epochs * train_records) / batch_size) - 1:
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
