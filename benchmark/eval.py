

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

slim = tf.contrib.slim

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


FLAGS = tf.app.flags.FLAGS


IMAGE_SIZE = 299
NUM_CLASSES = 1001


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.
  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    #if central_fraction:
    #  image = tf.image.central_crop(image, central_fraction=central_fraction)

    image = tf.reshape(tensor=image,shape=[height, width, 3])

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    #image = tf.multiply(image, 1/255.0)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


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

'''
def step_target_class_adversarial_images(x, eps, one_hot_target_class):
  """Base code for one step towards target class methods.

  Args:
    x: source images
    eps: size of adversarial perturbation
    one_hot_target_class: one hot encoded target classes for all images

  Returns:
    tensor with adversarial images
  """
  logits, end_points = create_model(x, reuse=True)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                  logits,
                                                  label_smoothing=0.1,
                                                  weights=1.0)
  cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                   end_points['AuxLogits'],
                                                   label_smoothing=0.1,
                                                   weights=0.4)
  x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
  x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
  return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x, eps):
  """One step towards least likely class (Step L.L.) adversarial examples.

  This method is an alternative to FGSM which does not use true classes.
  Method is described in the "Adversarial Machine Learning at Scale" paper,
  https://arxiv.org/abs/1611.01236

  Args:
    x: source images
    eps: size of adversarial perturbation

  Returns:
    adversarial images
  """
  logits, _ = create_model(x, reuse=True)
  least_likely_class = tf.argmin(logits, 1)
  one_hot_ll_class = tf.one_hot(least_likely_class, NUM_CLASSES)
  return step_target_class_adversarial_images(x, eps, one_hot_ll_class)


def stepllnoise_adversarial_images(x, eps):
  """Step L.L. with noise method.

  This is an imporvement of Step L.L. method. This method is better against
  adversarially trained models which learn to mask gradient.
  Method is described in the section "New randomized one shot attack" of
  "Ensemble Adversarial Training: Attacks and Defenses" paper,
  https://arxiv.org/abs/1705.07204

  Args:
    x: source images
    eps: size of adversarial perturbation

  Returns:
    adversarial images
  """
  logits, _ = create_model(x, reuse=True)
  least_likely_class = tf.argmin(logits, 1)
  one_hot_ll_class = tf.one_hot(least_likely_class, NUM_CLASSES)
  x_noise = x + eps / 2 * tf.sign(tf.random_normal(x.shape))
  return step_target_class_adversarial_images(x_noise, eps / 2,
                                              one_hot_ll_class)


def get_input_images(dataset_images):
  """Gets input images for the evaluation.

  Args:
    dataset_images: tensor with dataset images

  Returns:
    tensor with input images, which is either dataset images or adversarial
    images.

  Raises:
    ValueError: if adversarial method specified by --adversarial_method flag
      is invalid.
  """
  # adversarial_eps defines max difference of values of pixels if
  # pixels are in range [0, 255]. However values of dataset pixels are
  # in range [-1, 1], so converting epsilon.
  eps = FLAGS.adversarial_eps / 255 * 2.0

  if FLAGS.adversarial_method == 'stepll':
    return stepll_adversarial_images(dataset_images, eps)
  elif FLAGS.adversarial_method == 'stepllnoise':
    return stepllnoise_adversarial_images(dataset_images, eps)
  elif FLAGS.adversarial_method == 'none':
    return dataset_images
  else:
    raise ValueError('Invalid adversarial method: %s'
                     % (FLAGS.adversarial_method))


def _parse_data(example_proto):
    parsed_features=tf.parse_single_example(
        serialized=example_proto,
        features={
            "image_raw":tf.FixedLenFeature(shape=(),dtype=tf.string),
            "label":tf.FixedLenFeature(shape=(),dtype=tf.int64),
            "target_label":tf.FixedLenFeature(shape=(),dtype=tf.int64)
        }
    )

    # get single feature
    raw = parsed_features["image_raw"]
    label = parsed_features["label"]
    # decode raw
    image = tf.decode_raw(bytes=raw, out_type=tf.int64)
    return image,label
'''

def get_dataset(records_file, sample_num):
  #dataset=tf.data.TFRecordDataset(filenames=[records_file])
  #dataset=dataset.map(map_func=_parse_data)
  reader = tf.TFRecordReader
  file_name = records_file
  keys_to_feature = {
    "img_raw":tf.FixedLenFeature(shape=(),dtype=tf.string),
    "image_format":tf.FixedLenFeature((), tf.string, default_value='raw'),
    "label":tf.FixedLenFeature(shape=(),dtype=tf.int64),
    "target_label":tf.FixedLenFeature(shape=(),dtype=tf.int64)
  }
  items_to_handlers = {
    'image': slim.tfexample_decoder.Image(image_key='img_raw', format_key='image_format'),
    'label': slim.tfexample_decoder.Tensor('label'),
    'target_label': slim.tfexample_decoder.Tensor('target_label')
  }
  decoder=slim.tfexample_decoder.TFExampleDecoder(keys_to_feature, items_to_handlers)


  _ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'target_label': 'The target label.'
  }


  return slim.dataset.Dataset(
    data_sources=file_name,reader=reader,decoder=decoder,
    num_samples=sample_num, items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
    num_classes=NUM_CLASSES)


def main(_):
  #if not FLAGS.dataset_dir:
  #  raise ValueError('You must supply the dataset directory with --dataset_dir')

  sample_num = 1000
  records_file = FLAGS.test_tfrecords

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = tf.train.get_or_create_global_step()

    ###################
    # Prepare dataset #
    ###################
    #dataset = imagenet.get_split(FLAGS.split_name, FLAGS.dataset_dir)
    dataset = get_dataset(records_file, sample_num)

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)

    [dataset_image, label] = provider.get(['image', 'label'])
    dataset_image = preprocess_for_eval(dataset_image, IMAGE_SIZE, IMAGE_SIZE)
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

    labels = tf.squeeze(labels)
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