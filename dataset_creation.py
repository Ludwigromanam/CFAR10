import numpy as np
import cPickle as pickle
import tarfile
import tensorflow as tf

# Define global tensorflow variables and assign to FLAGS
tf.app.flags.DEFINE_integer('input_image_size', 32, 'The size of the images in the database')
tf.app.flags.DEFINE_integer('output_image_size', 24, 'The size of the cropped images for deep learning')
tf.app.flags.DEFINE_integer('input_image_channels', 3, 'The depth of the images in the database')
tf.app.flags.DEFINE_integer('num_labels', 10, 'The number of classes in the label file')
tf.app.flags.DEFINE_integer('cross_valid', 5000, 'The number of records to hold for cross-validation')

FLAGS = tf.app.flags.FLAGS


def extract(filename):
    tar = tarfile.open(filename)
    tar.extractall('./data/')
    tar.close()


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict


def randomize(dataset, labels):
    """
    Function to create a random split of the supplied data and labels. Used so that slices of the data will
    not be biased towards one class.
    :param dataset: Supplied image data where the first dimension is the image
    :param labels: The label of the image (same length as first dimension of dataset)
    :return: Shuffled datasets and labels
    """
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def tensorflow_conversion(images, labels, name):
    """
    Convert a numpy array to tensorflow binary format. It helps speed up reading data for training.
    :param images: image numpy array
    :param labels: label list
    :param name: naming convention for the file
    :return: Writes a binary file to be consumed by the read_data.py file
    """
    num_examples = labels.shape[0]
    filename = name + '.tfrecords'
    print 'Writing', filename
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())


def create_cfar10_data():
    # Extract, load and combine the raw CIFAR data.
    extract('./cifar-10-python.tar.gz')
    batch1 = unpickle('data/cifar-10-batches-py/data_batch_1')
    batch2 = unpickle('data/cifar-10-batches-py/data_batch_2')
    batch3 = unpickle('data/cifar-10-batches-py/data_batch_3')
    batch4 = unpickle('data/cifar-10-batches-py/data_batch_4')
    batch5 = unpickle('data/cifar-10-batches-py/data_batch_5')
    test_batch = unpickle('data/cifar-10-batches-py/test_batch')

    main_dataset = np.vstack([batch1['data'], batch2['data'], batch3['data'], batch4['data'], batch5['data']])
    main_labels = np.reshape(np.array(np.hstack([batch1['labels'], batch2['labels'],
                                      batch3['labels'], batch4['labels'], batch5['labels']])),
                             (np.shape(main_dataset)[0],))

    test_dataset = test_batch['data']
    test_labels = np.reshape(np.array(test_batch['labels']), (np.shape(test_dataset)[0],))

    # Randomize the data and then split the main dataset to train and validation datasets.
    main_dataset, main_labels = randomize(main_dataset, main_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)

    train_dataset = main_dataset[:np.shape(main_dataset)[0] - FLAGS.cross_valid, :]
    valid_dataset = main_dataset[np.shape(main_dataset)[0] - FLAGS.cross_valid:, :]

    train_labels = main_labels[:len(main_labels) - FLAGS.cross_valid]
    valid_labels = main_labels[len(main_labels) - FLAGS.cross_valid:]

    # Observe the data splits to make sure there isnt bias in the number of records
    print np.unique(valid_labels)
    print np.bincount(valid_labels)

    print np.unique(train_labels)
    print np.bincount(train_labels)

    print np.unique(test_labels)
    print np.bincount(test_labels)

    # The CIFAR data has a weird format... Need to make it RGB based images
    def reformat(dataset, labels):
        rgb = dataset.reshape(-1, FLAGS.input_image_channels, FLAGS.input_image_size * FLAGS.input_image_size)
        rgb = rgb.reshape(-1, FLAGS.input_image_channels, FLAGS.input_image_size, FLAGS.input_image_size)
        dataset = rgb.swapaxes(3, 1).swapaxes(1, 2)
        # labels = (np.arange(FLAGS.num_labels) == labels[:, None]).astype(np.int32)
        return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    print "Train Dataset Dimensions:"
    print np.shape(train_dataset), np.shape(train_labels)

    print "Valid Dataset Dimensions:"
    print np.shape(valid_dataset), np.shape(valid_labels)

    print "Test Dataset Dimensions:"
    print np.shape(test_dataset), np.shape(test_labels)

    # Convert and save the data in tensorflow format
    tensorflow_conversion(train_dataset, train_labels, 'train')
    tensorflow_conversion(valid_dataset, valid_labels, 'valid')
    tensorflow_conversion(test_dataset, test_labels, 'test')

