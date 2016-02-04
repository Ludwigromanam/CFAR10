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
        save_path = saver.save(session, "./model.ckpt")
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
    saver.restore(session, './model.ckpt')
    print "Model Restored"
    for step in xrange(int(num_epochs * (train_labels.shape[0]/batch_size)) + 1):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels,
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
        save_path = saver.save(session, "./model_refine.ckpt")
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
patch_size = 5
depth = 64
layer1 = 384
layer2 = 192

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_cfar10_data()

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  hidden_dprob = tf.placeholder('float')

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

  # Model.
  def train_model(data):
    conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv + b1)
    relu_dropout = tf.nn.dropout(relu, hidden_dprob)
    pool = tf.nn.max_pool(relu_dropout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(pool, w2, [1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv + b2)
    relu_dropout = tf.nn.dropout(relu, hidden_dprob)
    pool = tf.nn.max_pool(relu_dropout, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(tf.conv2d(pool, w3, [1, 1, 1, 1], padding='VALID') + b3)
    hidden_dropout = tf.nn.dropout(hidden, hidden_dprob)
    hidden2 = tf.nn.relu(tf.conv2d(hidden_dropout, w4, [1, 1, 1, 1], padding='VALID') + b4)
    hidden2_dropout = tf.nn.dropout(hidden2, hidden_dprob)
    output = tf.nn.conv2d(hidden2_dropout, w5, [1, 1, 1, 1], padding='VALID') + b5
    return tf.reshape(output, [-1, num_labels])

  def test_model(data):
    conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv + b1)
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(pool, w2, [1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(conv + b2)
    pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(tf.conv2d(pool, w3, [1, 1, 1, 1], padding='VALID') + b3)
    hidden2 = tf.nn.relu(tf.conv2d(hidden, w4, [1, 1, 1, 1], padding='VALID') + b4)
    output = tf.nn.conv2d(hidden2, w5, [1, 1, 1, 1], padding='VALID') + b5
    return tf.reshape(output, [-1, num_labels])


  # Training computation.
  logits = train_model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  #loss += 0.01 * (tf.nn.l2_loss(w3) + tf.nn.l2_loss(b3) +
  #               tf.nn.l2_loss(w4) + tf.nn.l2_loss(b4))

  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(0.05,
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
