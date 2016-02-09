import numpy as np
import cPickle as pickle
import tarfile
import tensorflow as tf

IN_IMAGE_HEIGHT = 32
IN_IMAGE_WIDTH = 32
OUT_IMAGE_HEIGHT = 24
OUT_IMAGE_WIDTH = 24

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
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def train_transform(dataset):
    new_dataset = np.empty((np.shape(dataset)[0], OUT_IMAGE_HEIGHT, OUT_IMAGE_WIDTH, 3))
    for image_index in range(0, np.shape(dataset)[0]):
        graph = tf.Graph()
        with graph.as_default():
            tf_image = tf.placeholder(tf.float32, (IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, 3))
            distorted_image = tf.image.random_crop(tf_image, [OUT_IMAGE_HEIGHT, OUT_IMAGE_WIDTH])
            distorted_image = tf.image.random_flip_left_right(distorted_image)
            distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
            white_image = tf.image.per_image_whitening(distorted_image)
        with tf.Session(graph=graph):
            image = dataset[image_index, :, :, :]
            transformed_image = white_image.eval(feed_dict={tf_image: image})
            new_dataset[image_index, :, :, :] = transformed_image
            if image_index % 100 == 0:
                print 'Now on record', image_index

    return new_dataset


def test_transform(dataset):
    new_dataset = np.empty((np.shape(dataset)[0], OUT_IMAGE_HEIGHT, OUT_IMAGE_WIDTH, 3))
    for image_index in range(0, np.shape(dataset)[0]):
        graph = tf.Graph()
        with graph.as_default():
            tf_image = tf.placeholder(tf.float32, (IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, 3))
            distorted_image = tf.image.resize_image_with_crop_or_pad(tf_image, OUT_IMAGE_HEIGHT, OUT_IMAGE_WIDTH)
            white_image = tf.image.per_image_whitening(distorted_image)
        with tf.Session(graph=graph):
            image = dataset[image_index, :, :, :]
            transformed_image = white_image.eval(feed_dict={tf_image: image})
            new_dataset[image_index, :, :, :] = transformed_image
            if image_index % 100 == 0:
                print 'Now on record', image_index

    return new_dataset


def create_cfar10_data():
    extract('./cifar-10-python.tar.gz')
    batch1 = unpickle('data/cifar-10-batches-py/data_batch_1')
    batch2 = unpickle('data/cifar-10-batches-py/data_batch_2')
    batch3 = unpickle('data/cifar-10-batches-py/data_batch_3')
    batch4 = unpickle('data/cifar-10-batches-py/data_batch_4')
    batch5 = unpickle('data/cifar-10-batches-py/data_batch_5')
    test_batch = unpickle('data/cifar-10-batches-py/test_batch')

    main_dataset = np.vstack([batch1['data'], batch2['data'], batch3['data'], batch4['data'], batch5['data']])
    main_labels = np.reshape(np.array(np.hstack([batch1['labels'], batch2['labels'],
                                      batch3['labels'], batch4['labels'], batch5['labels']])), (np.shape(main_dataset)[0],))

    test_dataset = test_batch['data']
    test_labels = np.reshape(np.array(test_batch['labels']), (np.shape(test_dataset)[0],))

    main_dataset, main_labels = randomize(main_dataset, main_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)

    cross_valid = 5000

    train_dataset = main_dataset[:np.shape(main_dataset)[0] - cross_valid, :]
    valid_dataset = main_dataset[np.shape(main_dataset)[0]-cross_valid:, :]

    train_labels = main_labels[:len(main_labels)-cross_valid]
    valid_labels = main_labels[len(main_labels)-cross_valid:]

    print np.unique(valid_labels)
    print np.bincount(valid_labels)

    print np.unique(train_labels)
    print np.bincount(train_labels)

    print np.unique(test_labels)
    print np.bincount(test_labels)

    image_size = 32
    num_channels = 3
    num_labels = 10

    def reformat(dataset, labels):
      rgb = dataset.reshape(-1, num_channels, image_size * image_size).reshape(-1, num_channels, image_size, image_size)
      dataset = rgb.swapaxes(3, 1).swapaxes(1, 2)
      labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
      return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    train_dataset = train_transform(train_dataset)
    print "Train Dataset Dimensions:"
    print np.shape(train_dataset), np.shape(train_labels)

    valid_dataset = test_transform(valid_dataset)
    print "Valid Dataset Dimensions:"
    print np.shape(valid_dataset), np.shape(valid_labels)

    test_dataset = test_transform(test_dataset)
    print "Test Dataset Dimensions:"
    print np.shape(test_dataset), np.shape(test_labels)

    pickle_file = 'CFAR10.pickle'

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print 'Unable to save data to', pickle_file, ':', e
        raise


def get_cfar10_data():
    pickle_file = 'CFAR10.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save

    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
