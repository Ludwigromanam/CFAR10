import numpy as np
import tensorflow as tf
from dataset_creation import get_cfar10_data


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def batch_accuracy_test(session, batch_size, dataset, labels, tensorflow_dataset, tensorflow_function):

  accurate_labels = 0.0

  for step in xrange(int(labels.shape[0]/batch_size)):
    offset = (step * batch_size) % (labels.shape[0] - batch_size)
    batch_data = dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = labels[offset:(offset + batch_size), :]
    feed_dict = {tensorflow_dataset: batch_data}
    predictions = session.run([tensorflow_function], feed_dict=feed_dict)
    predictions = np.reshape(np.array(predictions), (batch_size, labels.shape[1]))
    accurate_labels += np.sum(np.argmax(predictions, 1) == np.argmax(batch_labels, 1))

  return (100.0 * accurate_labels)/labels.shape[0]


def initial_model_session(graph,
                          num_epochs, batch_size,
                          train_dataset, train_labels):
  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print "Initialized"
    for step in xrange(int(num_epochs * (train_labels.shape[0]/batch_size)) + 1):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels,
                   conv_dprob: 0.7,
                   hidden_dprob: 0.7}
      _, l, lr, predictions = session.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
      if (step % 100 == 0 and step != 0):
        print "------------------------------------------"
        print "Current epoch: ", (float(step) * batch_size) / train_labels.shape[0]
        print "Current learning rate: ", lr
        print "Minibatch loss at step", step, ":", l
        print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
        print "Validation accuracy: %.1f%%" % batch_accuracy_test(session=session, batch_size=batch_size,
                                                                  dataset=valid_dataset, labels=valid_labels,
                                                                  tensorflow_dataset=tf_valid_dataset,
                                                                  tensorflow_function=valid_prediction)
      if (step % 1000 == 0 and step != 0):
        # Save the variables to disk.
        save_path = saver.save(session, "./model2.ckpt")
        print "Model saved in file: ", save_path

    print "===================================="
    print "Test accuracy: %.1f%%" % batch_accuracy_test(session=session, batch_size=batch_size,
                                                        dataset=test_dataset, labels=test_labels,
                                                        tensorflow_dataset=tf_test_dataset,
                                                        tensorflow_function=test_prediction)


def refine_model_session(graph,
                          num_epochs, batch_size,
                          train_dataset, train_labels):
  with tf.Session(graph=graph) as session:
    saver.restore(session, './model2.ckpt')
    print "Model Restored"
    for step in xrange(int(num_epochs * (train_labels.shape[0]/batch_size)) + 1):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels,
                   conv_dprob: 0.7,
                   hidden_dprob: 0.7}
      _, l, lr, predictions = session.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
      if (step % 100 == 0):
        print "------------------------------------------"
        print "Current epoch: ", (float(step) * batch_size) / train_labels.shape[0]
        print "Current learning rate: ", lr
        print "Minibatch loss at step", step, ":", l
        print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
        print "Validation accuracy: %.1f%%" % batch_accuracy_test(session=session, batch_size=batch_size,
                                                                  dataset=valid_dataset, labels=valid_labels,
                                                                  tensorflow_dataset=tf_valid_dataset,
                                                                  tensorflow_function=valid_prediction)
      if (step % 1000 == 0):
        # Save the variables to disk.
        save_path = saver.save(session, "./model2_refine.ckpt")
        print "Model saved in file: ", save_path

    print "===================================="
    print "Test accuracy: %.1f%%" % batch_accuracy_test(session=session, batch_size=batch_size,
                                                        dataset=test_dataset, labels=test_labels,
                                                        tensorflow_dataset=tf_test_dataset,
                                                        tensorflow_function=test_prediction)


image_size = 32
num_labels = 10
num_channels = 3 # grayscale
batch_size = 200
patch_size = 3
depth1 = 64
depth2 = 128
depth3 = 384
depth4 = 192

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_cfar10_data()

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  conv_dprob = tf.placeholder('float')
  hidden_dprob = tf.placeholder('float')

  # Variables.
  conv1_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth1], stddev=0.1))
  conv1_bias = tf.Variable(tf.zeros([depth1]))
  conv2_weight = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth1, depth1], stddev=0.1))
  conv2_bias = tf.Variable(tf.constant(1.0, shape=[depth1]))

  conv3_weight = tf.Variable(tf.truncated_normal([image_size/2, image_size/2, depth1, depth2], stddev=0.1))
  conv3_bias = tf.Variable(tf.constant(1.0, shape=[depth2]))
  conv4_weight = tf.Variable(tf.truncated_normal([image_size/2, image_size/2, depth2, depth2], stddev=0.1))
  conv4_bias = tf.Variable(tf.constant(1.0, shape=[depth2]))

  conv5_weight = tf.Variable(tf.truncated_normal([image_size/(2*2), image_size/(2*2), depth2, depth3], stddev=0.1))
  conv5_bias = tf.Variable(tf.constant(1.0, shape=[1, 1, depth3]))
  conv6_weight = tf.Variable(tf.truncated_normal([1, 1, depth3, depth4], stddev=0.1))
  conv6_bias = tf.Variable(tf.constant(1.0, shape=[1, 1, depth4]))

  conv7_weight = tf.Variable(tf.truncated_normal([1, 1, depth4, num_labels], stddev=0.1))
  conv7_bias = tf.Variable(tf.constant(1.0, shape=[1, 1, num_labels]))

  # Model.
  def train_model(data):
    conv1 = tf.nn.conv2d(data, conv1_weight, [1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_bias)
    relu1_dropout = tf.nn.dropout(relu1, conv_dprob)
    print relu1.get_shape().as_list()
    conv2 = tf.nn.conv2d(relu1_dropout, conv2_weight, [1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(conv2 + conv2_bias)
    relu2_dropout = tf.nn.dropout(relu2, conv_dprob)
    print relu2.get_shape().as_list()
    pool1 = tf.nn.max_pool(relu2_dropout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print pool1.get_shape().as_list()

    conv3 = tf.nn.conv2d(pool1, conv3_weight, [1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(conv3 + conv3_bias)
    relu3_dropout = tf.nn.dropout(relu3, conv_dprob)
    print relu3.get_shape().as_list()
    conv4 = tf.nn.conv2d(relu3_dropout, conv4_weight, [1, 1, 1, 1], padding='SAME')
    relu4 = tf.nn.relu(conv4 + conv4_bias)
    relu4_dropout = tf.nn.dropout(conv4, conv_dprob)
    print relu4.get_shape().as_list()
    pool2 = tf.nn.max_pool(relu4_dropout, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print pool2.get_shape().as_list()

    hidden_conv1 = tf.nn.conv2d(pool2, conv5_weight, [1, 1, 1, 1], padding='VALID')
    hidden_relu1 = tf.nn.relu(hidden_conv1 + conv5_bias)
    hidden_relu1_dropout = tf.nn.dropout(hidden_relu1, hidden_dprob)
    print hidden_relu1.get_shape().as_list()
    hidden_conv2 = tf.nn.conv2d(hidden_relu1_dropout, conv6_weight, [1, 1, 1, 1], padding='VALID')
    hidden_relu2 = tf.nn.relu(hidden_conv2 + conv6_bias)
    hidden_relu2_dropout = tf.nn.dropout(hidden_relu2, hidden_dprob)
    print hidden_relu2.get_shape().as_list()

    output = tf.nn.conv2d(hidden_relu2_dropout, conv7_weight, [1, 1, 1, 1], padding='VALID')
    print output.get_shape().as_list()
    output = tf.reshape(output + conv7_bias, [-1, num_labels])
    print output.get_shape().as_list()
    return output

  def test_model(data):
    conv1 = tf.nn.conv2d(data, conv1_weight, [1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_bias)
    conv2 = tf.nn.conv2d(relu1, conv2_weight, [1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(conv2 + conv2_bias)
    pool1 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv3 = tf.nn.conv2d(pool1, conv3_weight, [1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(conv3 + conv3_bias)
    conv4 = tf.nn.conv2d(relu3, conv4_weight, [1, 1, 1, 1], padding='SAME')
    relu4 = tf.nn.relu(conv4 + conv4_bias)
    pool2 = tf.nn.max_pool(relu4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    hidden_conv1 = tf.nn.conv2d(pool2, conv5_weight, [1, 1, 1, 1], padding='VALID')
    hidden_relu1 = tf.nn.relu(hidden_conv1 + conv5_bias)
    hidden_conv2 = tf.nn.conv2d(hidden_relu1, conv6_weight, [1, 1, 1, 1], padding='VALID')
    hidden_relu2 = tf.nn.relu(hidden_conv2 + conv6_bias)

    output = tf.nn.conv2d(hidden_relu2, conv7_weight, [1, 1, 1, 1], padding='VALID')
    output = tf.reshape(output + conv7_bias, [-1, num_labels])
    return output


  # Training computation.
  logits = train_model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  #loss += 0.01 * (tf.nn.l2_loss(w3) + tf.nn.l2_loss(b3) +
  #               tf.nn.l2_loss(w4) + tf.nn.l2_loss(b4))

  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(0.0005,
                                            global_step * batch_size,
                                            train_labels.shape[0] * 10,
                                            0.99,
                                            staircase=True)
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(test_model(tf_train_dataset))
  valid_prediction = tf.nn.softmax(test_model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(test_model(tf_test_dataset))


num_epochs = 300
initial_model_session(graph=graph, num_epochs=num_epochs, batch_size=batch_size, train_dataset=train_dataset,
                     train_labels=train_labels)