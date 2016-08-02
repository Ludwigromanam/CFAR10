import tensorflow as tf

# Some important variables
input_image_size = 32
output_image_size = 24
input_image_channels = 3
num_labels = 10
valid_records = 5000
test_records = 10000
train_records = 45000
batch_size = 100


def distorted_inputs(num_threads):
    """
    For training the data, its important to transform the image to create a level of variance that might be
    expected in the real world. Some of the things that are noted here are:
        -Flipping the image
        -Turning the image
        -Tuning the brightness of the image
        -Tuning the contrast of the image
        -Whiten the image
    It might be a good project to create a code script that approximates these values per image in the train and test
    set. This can give a "realistic" range to use for a lot of the tuning parameters
    :param num_threads: Number of threads for batching
    :return:
    """
    # Create a queue from the tensorflow binary file and define
    filename_queue = tf.train.string_input_producer(['train.tfrecords'])
    result = read_data(filename_queue)
    min_queue = train_records * 0.4

    # Run image distortions as a way to artificially increase the size and expressiveness of the training data

    # The below code does the same thing on both lines, but the tensorflow version determines which to use
    distorted_image = tf.image.random_crop(result.image, [output_image_size, output_image_size])
    #distorted_image = tf.random_crop(result.image, [output_image_size, output_image_size,
    #                                                      input_image_channels])

    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    white_image = tf.image.per_image_whitening(distorted_image)

    return generate_batches(white_image, result.label, min_queue, num_threads)


def inputs(filename):
    """
    For inference/testing/validating the data. This does the base transforms to make the image compatible with
    the trained convnet model. It is used for inference, testing, and validation.
    :param filename: Tells the function where to point. Either the valid.tfrecords or test.tfrecords in this
    codebase
    :return:
    """
    filename_queue = tf.train.string_input_producer([filename])
    result = read_data(filename_queue)

    if filename == 'valid.tfrecords':
        min_queue = valid_records * 0.4
    else:
        min_queue = test_records * 0.4

    distorted_image = tf.image.resize_image_with_crop_or_pad(result.image, output_image_size,
                                                             output_image_size)
    white_image = tf.image.per_image_whitening(distorted_image)

    return generate_batches(white_image, result.label, min_queue, num_threads=1)


def read_data(filename_queue):
    """
    read_data is an access object to take a .tfrecord and transform it for modeling purposes. it hs both
    a label and an image associated with it
    :param filename_queue: The queue runner created by tensorflow
    :return: An object of the class CIFAR10Record that has both an label and an image value
    """
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        #dense_keys=['image_raw', 'label'],
        #dense_types=[tf.string, tf.int64]
        features={'image_raw': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)}
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([input_image_size * input_image_size * input_image_channels])
    image = tf.cast(image, tf.float32)
    result.image = tf.reshape(image, [input_image_size, input_image_size, input_image_channels])

    label = tf.cast(features['label'], tf.int32)
    result.label = tf.sparse_to_dense(label, [num_labels], 1.0, 0.0)

    return result


def generate_batches(image, label, min_queue_examples, num_threads):
    """
    Function to generate batches used by the model. The batches are important because they stage the data while
    it is being fed through the model and helps speed up processing. It creates a bucket that gets pulled
    by batch sizes to be processed by the model.
    :param image: The an image object
    :param label: The label of the image
    :param min_queue_examples: The number of items to queue
    :param num_threads: Number of threads to commit to creating the parsed data bucket
    :return:
    """
    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        capacity=int(min_queue_examples + 3 * batch_size),
        num_threads=num_threads,
        min_after_dequeue=int(min_queue_examples)
    )

    return images, labels