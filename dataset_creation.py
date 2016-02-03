import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import tarfile


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


def mean_center(dataset):
    return ((dataset - 255.0/2)/255.0).astype(np.float32)


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

    print np.shape(main_dataset)
    print np.shape(main_labels)

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

    print "Train Dataset Dimensions:"
    print np.shape(train_dataset), np.shape(train_labels)

    print "Valid Dataset Dimensions:"
    print np.shape(valid_dataset), np.shape(valid_labels)

    print "Test Dataset Dimensions:"
    print np.shape(test_dataset), np.shape(test_labels)

    image_size = 32
    num_channels = 3
    num_labels = 10

    def format_data(rgb, width=32, height=32, num_colors=3):
       rgb = rgb.reshape(-1, num_colors, width * height).reshape(-1, num_colors, width, height)
       return rgb.swapaxes(3, 1).swapaxes(1, 2)

    def reformat(dataset, labels):
      rgb = dataset.reshape(-1, num_channels, image_size * image_size).reshape(-1, num_channels, image_size, image_size)
      dataset = rgb.swapaxes(3, 1).swapaxes(1, 2)
      labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
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

    train_dataset = mean_center(train_dataset)
    valid_dataset = mean_center(valid_dataset)
    test_dataset = mean_center(test_dataset)

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