

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import sys
sys.path.insert(0, "/home/haojieyuan/autoAdv/benchmark/models/research/slim")

from nets import resnet_v2
from nets import inception
from nets import inception_resnet_v2
from nets import nets_factory
import pdb

slim = tf.contrib.slim
from preprocess import preprocess_for_eval

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'oxfordFlowers', 'oxfordFlowers, fgvcAircraft, stanfordCars.')

tf.app.flags.DEFINE_string(
    'test_tfrecords', '/home/haojieyuan/Data/ImageNet/nips2017dev.tfrecords',
    'Test tesorflow records file path. ')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3',
    'Name of the model to use, either "inception_v3" "inception_v4" "resnet_v2" or "inception_resnet_v2"')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_string(
    'adversarial_method', 'none',
    'What kind of adversarial examples to use for evaluation. '
    'Could be one of: "none", "stepll", "stepllnoise".')

tf.app.flags.DEFINE_float(
    'adversarial_eps', 0.0,
    'Size of adversarial perturbation in range [0, 255].')

tf.app.flags.DEFINE_integer(
    'num_classes', 0,
    'Size of adversarial perturbation in range [0, 255].')


FLAGS = tf.app.flags.FLAGS


IMAGE_SIZE = 299
NUM_CLASSES = FLAGS.num_classes

'''
def preprocess_for_eval(image, in_height, in_width, scope=None):
    # make it same as ImageNet
    out_height = 299
    out_width = 299
    with tf.name_scope(scope, 'eval_image', [image, in_height, in_width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.reshape(tensor=image, shape=[in_height, in_width, 3])

    if out_height and out_width:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [out_height, out_width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image
'''

def create_model(x, reuse=None):
  """Create model graph.

  Args:
    x: input images
    reuse: reuse parameter which will be passed to underlying variable scopes.
      Should be None first call and True every subsequent call.

  Returns:
    (logits, end_points) - tuple of model logits and enpoints

  Raises:
    ValueError: if model type specified by --model_name flag is invalid.
  """
  if FLAGS.model_name == 'inception_v3':
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      return inception.inception_v3(
          x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
  elif FLAGS.model_name == 'inception_v4':
    with slim.arg_scope(inception.inception_v4_arg_scope()):
      return inception.inception_v4(
          x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
  elif FLAGS.model_name == 'inception_resnet_v2':
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      return inception_resnet_v2.inception_resnet_v2(
          x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
  elif FLAGS.model_name == 'resnet_v2':
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      return resnet_v2.resnet_v2_152(
          x, num_classes=NUM_CLASSES, is_training=False, reuse=reuse)
  else:
    raise ValueError('Invalid model name: %s' % (FLAGS.model_name))



def get_dataset(dataset_name, partition):
    if dataset_name == 'oxfordFlowers':
        dataset_prefix = '/home/haojieyuan/Data/OxfordFlowers'
        sample_num = 2040 if partition=='train' else 6149

    elif dataset_name == 'fgvcAircraft':
        dataset_prefix = '/home/haojieyuan/Data/FGVC_Aircraft'
        sample_num = 6667 if partition=='train' else 3333

    elif dataset_name == 'stanfordCars':
        dataset_prefix = '/home/haojieyuan/Data/stanfordCars'
        sample_num = 8144 if partition=='train' else 8041

    else:
        assert False, 'Unknown dataset name.'

    file_name = os.path.join(dataset_prefix, partition+'.tfrecords')
    reader = tf.TFRecordReader
    keys_to_feature = {
        "img_raw":tf.FixedLenFeature(shape=(),dtype=tf.string),
        "image_format":tf.FixedLenFeature((), tf.string, default_value='raw'),
        "label":tf.FixedLenFeature(shape=(),dtype=tf.int64),
        "height":tf.FixedLenFeature(shape=(),dtype=tf.int64),
        "width":tf.FixedLenFeature(shape=(),dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(image_key='img_raw', format_key='image_format'),
        'label': slim.tfexample_decoder.Tensor('label'),
        'height': slim.tfexample_decoder.Tensor('height'),
        'width': slim.tfexample_decoder.Tensor('width'),
    }
    decoder=slim.tfexample_decoder.TFExampleDecoder(keys_to_feature, items_to_handlers)

    _ITEMS_TO_DESCRIPTIONS = {
        'image':  'A color image of varying height and width.',
        'label':  'The label id of the image, interger between 0 and class_num-1.',
        'height': 'Img height',
        'width':  'img width'
    }

    return slim.dataset.Dataset(
        data_sources=file_name,reader=reader,decoder=decoder,
        num_samples=sample_num, items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=NUM_CLASSES)


def main(_):
  #if not FLAGS.dataset_dir:
  #  raise ValueError('You must supply the dataset directory with --dataset_dir')

  records_file = FLAGS.test_tfrecords

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = tf.train.get_or_create_global_step()

    ###################
    # Prepare dataset #
    ###################
    #dataset = imagenet.get_split(FLAGS.split_name, FLAGS.dataset_dir)
    dataset = get_dataset(FLAGS.dataset_name, 'train')

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)

    [dataset_image, label, im_h, im_w] = provider.get(['image', 'label', 'height', 'width'])
    #dataset_image = preprocess_for_eval(dataset_image, im_h, im_w)
    dataset_image = slim.preprocessing.inception_preprocessing.preprocess_for_eval(
      dataset_image, im_h, im_w)

    #label = tf.constant(1, dtype=tf.int64)
    dataset_images, labels = tf.train.batch(
        [dataset_image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ########################################
    # Define the model and input exampeles #
    ########################################

    create_model(tf.placeholder(tf.float32, shape=dataset_images.shape))
    #input_images = get_input_images(dataset_images)
    input_images = dataset_images
    logits, _ = create_model(input_images, reuse=True)

    '''
    network_fn = nets_factory.get_network_fn(
            'inception_v3',
            num_classes=NUM_CLASSES,
            is_training=False)

    logits, end_points = network_fn(dataset_images)
    '''


    if FLAGS.moving_average_decay > 0:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    ######################
    # Define the metrics #
    ######################
    predictions = tf.argmax(logits, 1)

    # other wise predictions will be (B, 1, 1001)
    #if FLAGS.model_name == 'resnet_v2':
    #  predictions = tf.squeeze(predictions)


    #labels = tf.squeeze(labels)
    #labels = slim.one_hot_encoding(labels, NUM_CLASSES)
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_sparse_recall_at_k(
            logits, tf.reshape(labels, [-1, 1]), 5),
    })

    ######################
    # Run evaluation     #
    ######################
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    # Not Useful.
    #slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore,
    #                               ignore_missing_vars=False)

    top1_accuracy, top5_accuracy = slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=None,
        summary_op=None,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        final_op=[names_to_values['Accuracy'], names_to_values['Recall_5']],
        variables_to_restore=variables_to_restore)

    print('Top1 Accuracy: ', top1_accuracy)
    print('Top5 Accuracy: ', top5_accuracy)


if __name__ == '__main__':
  tf.app.run()