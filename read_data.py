import tensorflow as tf

tf.app.flags.DEFINE_integer('input_image_size', 32, 'The size of the images in the database')
tf.app.flags.DEFINE_integer('output_image_size', 24, 'The size of the cropped images for deep learning')
tf.app.flags.DEFINE_integer('input_image_channels', 3, 'The depth of the images in the database')
tf.app.flags.DEFINE_integer('num_labels', 10, 'The number of classes in the label file')
tf.app.flags.DEFINE_integer('valid_records', 5000, 'The number of validations records')
tf.app.flags.DEFINE_integer('test_records', 10000, 'The number of validations records')
tf.app.flags.DEFINE_integer('train_records', 45000, 'The number of validations records')
tf.app.flags.DEFINE_integer('batch_size', 100, 'The batch size.')

FLAGS = tf.app.flags.FLAGS


def distorted_inputs(num_epochs):

    filename_queue = tf.train.string_input_producer(['train.tfrecords'], num_epochs=num_epochs)
    result = read_data(filename_queue)
    min_queue = FLAGS.train_records * 0.4

    distorted_image = tf.image.random_crop(result.image, [FLAGS.output_image_size, FLAGS.output_image_size])
    #distorted_image = tf.image.random_crop(result.image, [FLAGS.output_image_size, FLAGS.output_image_size,
    #                                                      FLAGS.input_image_channels])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    white_image = tf.image.per_image_whitening(distorted_image)

    return generate_batches(white_image, result.label, min_queue)


def inputs(filename):

    filename_queue = tf.train.string_input_producer([filename])
    result = read_data(filename_queue)

    if filename == 'valid.tfrecords':
        min_queue = FLAGS.valid_records * 0.4
    else:
        min_queue = FLAGS.test_records * 0.4

    distorted_image = tf.image.resize_image_with_crop_or_pad(result.image, FLAGS.output_image_size,
                                                             FLAGS.output_image_size)
    white_image = tf.image.per_image_whitening(distorted_image)

    return generate_batches(white_image, result.label, min_queue)


def read_data(filename_queue):

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        dense_keys=['image_raw', 'label'],
        dense_types=[tf.string, tf.int64]
        #features={'image_raw': tf.FixedLenFeature([], tf.string),
        #          'label': tf.FixedLenFeature([], tf.int64)}
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([FLAGS.input_image_size * FLAGS.input_image_size * FLAGS.input_image_channels])
    image = tf.cast(image, tf.float32)
    result.image = tf.reshape(image, [FLAGS.input_image_size, FLAGS.input_image_size, FLAGS.input_image_channels])

    label = tf.cast(features['label'], tf.int32)
    result.label = tf.sparse_to_dense(label, [FLAGS.num_labels], 1.0, 0.0)

    return result


def generate_batches(image, label, min_queue_examples):

    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=FLAGS.batch_size,
        capacity=int(min_queue_examples + 3 * FLAGS.batch_size),
        num_threads=5,
        min_after_dequeue=int(min_queue_examples)
    )

    return images, labels