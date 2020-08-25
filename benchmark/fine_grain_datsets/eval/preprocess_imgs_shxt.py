"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
start_time = time.time()

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import pdb
import sys
sys.path.insert(0, '/home/haojieyuan/autoAdv/benchmark/attacks/TI')
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

slim = tf.contrib.slim



DEBUG_ = False



tf.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string('checkpoint_path_inception_v3', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_v4', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_resnet_v2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('target_model', '', 'Choosen target model: ens, resnet, inception_v3, inception_v4, inception_resnet_v2')

tf.flags.DEFINE_string('input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon', 32.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 2, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_bool('use_ti', False, 'Use translation invariant or not.')

tf.flags.DEFINE_bool('use_si', False, 'Use scale invariant or not.')

tf.flags.DEFINE_float('prob', 0.4, 'probability of using diverse inputs.')

tf.flags.DEFINE_string('autoaug_file', '', 'auto augmentation search result.')



FLAGS = tf.flags.FLAGS

if FLAGS.prob !=0 :
    print("Using DI.")

if FLAGS.use_ti:
    print("Using TI.")

if FLAGS.use_si:
    print("Using SI.")
    SI_weights = FLAGS.batch_size*[1.0, 1.0, 1.0, 1.0, 1.0]
    SI_weights_04 = FLAGS.batch_size*[0.4, 0.4, 0.4, 0.4, 0.4]

if os.path.exists(FLAGS.autoaug_file):
    print("Using auto augment.")
    USE_AUTO_AUG = True
    with open(FLAGS.autoaug_file) as f:
        AUG_POLICY = eval(f.readline())
    #AUG_weights = [aug_weight for aug_type, aug_weight, aug_prob, aug_range in AUG_POLICY]
    AUG_weights = [branch[0] for branch in AUG_POLICY]
    w_sum = sum(AUG_weights)
    AUG_weights = [aug_weight/w_sum for aug_weight in AUG_weights]

    AUG_weights_1 = FLAGS.batch_size*AUG_weights
    AUG_weights_04 = [weight_*0.4 for weight_ in AUG_weights_1]

    AUG_num = len(AUG_POLICY)
else:
    USE_AUTO_AUG = False

assert not(FLAGS.use_si and USE_AUTO_AUG), "Using both auto aug and scale invariant is not supported yet."


AUG_TYPE = {0: 'resize_padding', 1: 'translation', 2: 'rotation',
            3: 'gaussian_noise', 4: 'horizontal_flip', 5: 'vertical_flip',
            6: 'scaling', 7: 'invert', 8: 'solarize'}


def augmentation(type, prob, mag_range, input_tensor):

    op_type = AUG_TYPE[type]
    mag_range = int(mag_range)

    if mag_range == 0 or prob == 0:
        return input_tensor

    if op_type == 'resize_padding':
        mag = tf.random_uniform((), 0, mag_range, dtype=tf.int32)
        mag = tf.cast(mag, tf.float32)
        w_modified = 2*(1 + tf.cast(0.01*mag*FLAGS.image_width, tf.int32))
        h_modified = 2*(1 + tf.cast(0.01*mag*FLAGS.image_width, tf.int32))
        w_resized = FLAGS.image_width - w_modified
        h_resized = FLAGS.image_width - h_modified

        rescaled = tf.image.resize_images(input_tensor, [w_resized, h_resized],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        h_padding_t = tf.random_uniform((), 0, h_modified, dtype=tf.int32)
        h_padding_b = h_modified - h_padding_t
        w_padding_l = tf.random_uniform((), 0, w_modified, dtype=tf.int32)
        w_padding_r = w_modified - w_padding_l

        padded = tf.pad(rescaled, [[0, 0], [h_padding_t, h_padding_b],
                                           [w_padding_l, w_padding_r], [0, 0]], constant_values=0.)
        padded.set_shape((input_tensor.shape[0], FLAGS.image_width, FLAGS.image_width, 3))
        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda: padded, lambda: input_tensor)

    elif op_type == 'translation':
        r1 = tf.random_uniform(shape=[1])[0]
        r2 = tf.random_uniform(shape=[1])[0]
        w_direction = tf.math.sign(r1-0.5)
        h_direction = tf.math.sign(r2-0.5)

        mag = tf.random_uniform((), 0, mag_range, dtype=tf.int32)
        mag = tf.cast(mag, tf.float32)
        w_modified = w_direction*0.02*mag*FLAGS.image_width
        h_modified = h_direction*0.02*mag*FLAGS.image_width

        w_modified = tf.cast(w_modified, tf.int32)
        h_modified = tf.cast(h_modified, tf.int32)
        w_modified = tf.cast(w_modified, tf.float32)
        h_modified = tf.cast(h_modified, tf.float32)

        translated = tf.contrib.image.translate(input_tensor, [w_modified, h_modified])
        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda:translated, lambda: input_tensor)

    elif op_type == 'rotation':
        r1 = tf.random_uniform(shape=[1])[0]
        rotate_direction = tf.math.sign(r1-0.5)
        mag = tf.random_uniform((), 0, mag_range, dtype=tf.int32)
        mag = tf.cast(mag, tf.float32)

        rotate_reg = rotate_direction*math.pi*mag
        rotated = tf.contrib.image.rotate(input_tensor, rotate_reg)
        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda:rotated, lambda: input_tensor)

    elif op_type == 'gaussian_noise':
        mag = tf.random_uniform((), 0, mag_range, dtype=tf.int32)
        mag = tf.cast(mag, tf.float32)
        # image tensor here ranges [-1, 1]
        input_tensor = (input_tensor + 1.0) * 0.5  # now [0, 1]
        rnd = tf.random_uniform(shape=input_tensor.shape)
        noised = input_tensor + rnd * mag/60
        noised = tf.clip_by_value(noised, 0, 1.0)
        noised = noised * 2.0 - 1.0  # [-1, 1] again
        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda:noised, lambda: input_tensor)

    elif op_type == 'horizontal_flip':
        h_fliped = tf.image.flip_left_right(input_tensor)
        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda:h_fliped, lambda: input_tensor)

    elif op_type == 'vertical_flip':
        v_fliped = tf.image.flip_up_down(input_tensor)
        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda:v_fliped, lambda: input_tensor)

    elif op_type == 'scaling':
        mag = tf.random_uniform((), 0, mag_range, dtype=tf.int32)
        mag = tf.cast(mag, tf.float32)
        scaling_factor = 1.0 - 0.1*mag # 1.0~0.1
        scaled_tensor = scaling_factor*input_tensor
        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda:scaled_tensor, lambda: input_tensor)

    elif op_type == 'invert':
        inverted = 1.0 - input_tensor
        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda:inverted, lambda: input_tensor)

    elif op_type == 'solarize':
        mag = tf.random_uniform((), 0, mag_range, dtype=tf.int32)
        mag = tf.cast(mag, tf.float32)
        input_tensor = (input_tensor + 1.0)/2.0 # -1~1 to 0~1
        solarize_threshold = 1.0 - 0.09*mag
        mask = tf.greater(input_tensor, solarize_threshold)
        mask = tf.cast(mask, tf.float32)

        solarized = input_tensor*(1.0-mask) + (1.0-input_tensor)*mask # invert those above thres
        solarized = solarized*2.0 - 1.0  # 0~1 to -1~1

        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda:solarized, lambda: input_tensor)



    elif op_type == 'equalize':
        input_tensor = (input_tensor + 1.0)/2.0 * 255.0 # -1~1 to 0~255

        # batch dim iterate
        out = []
        for i in range(input_tensor.shape[0]):
            img = input_tensor[i,:,:,:] # NHWC -> HWC
            #img = tf.expand_dims(img, 0)

            eqalized_img = []

            for j in range(img.shape[-1]):
                channel_j = img[:,:,j]  # HWC -> HW
                channel_j = tf.expand_dims(channel_j, -1) # HW -> HWC
                value_range = tf.constant([0., 255.], dtype=tf.float32)
                histogram = tf.histogram_fixed_width(channel_j, value_range, 256)
                cdf = tf.cumsum(histogram)
                cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

                img_shape = tf.shape(channel_j)
                pix_cnt = img_shape[-3] * img_shape[-2]
                px_map = tf.round(tf.to_float(cdf - cdf_min) * 255. / tf.to_float(pix_cnt-1))
                px_map = tf.cast(px_map, tf.uint8)

                eq_hist = tf.expand_dims(tf.gather_nd(px_mp, tf.cast(channel_j, tf.int32)), 2)

                equalized_img.append(eq_hist)


            out.append(tf.concat(equalized_img, -1)) #[[HWC], [HWC], [HWC]]


        out_batch = tf.stack(out) # [] -> NHWC
        out_batch = (out_batch/255.0)*2.0 - 1.0 # 0~255 -> -1~1

        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.1*prob), lambda:out_batch, lambda: input_tensor)



def autoaug_diversity(input_tensor):
    #auged_list = [augmentation(aug_type, aug_prob, aug_range, input_tensor) for \
    #                           aug_type, aug_weight, aug_prob, aug_range in AUG_POLICY]
    auged_list = [branch_augmentation(input_tensor, branch_policy[1:]) for \
                                      branch_policy in AUG_POLICY]

    return tf.concat(auged_list, 0) # concat on 0 axis

def branch_augmentation(x, branch_policy):
    for aug_type, aug_prob, aug_range in branch_policy:
        x = augmentation(aug_type, aug_prob, aug_range, x)

    return x


def si_diversity(input_tensor):
    scale_list = [1.0, 1.0/2.0, 1.0/4.0, 1.0/8.0, 1.0/16.0]
    si_list = [input_tensor*scale_factor for scale_factor in scale_list]

    return tf.concat(si_list, 0)


def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

# 7 for normal models, 15 for adv trained models.
kernel = gkern(15, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)

def preprocess_for_eval(image, in_height=None, in_width=None,
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
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  out_height = 299
  out_width = 299
  with tf.name_scope(scope, 'eval_image', [image, in_height, in_width]):
    #if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  #image = tf.reshape(tensor=image, shape=[in_height, in_width, 3])

  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  if central_fraction:
    image = tf.image.central_crop(image, central_fraction=central_fraction)

  if out_height and out_width:
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_image_with_pad(image, out_height, out_width)
    image = tf.squeeze(image, [0])

  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image

def preprocess_for_eval_back(image, in_height=None, in_width=None,
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
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  out_height = 299
  out_width = 299
  with tf.name_scope(scope, 'eval_image', [image, in_height, in_width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  #image = tf.reshape(tensor=image, shape=[in_height, in_width, 3])

  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  if central_fraction:
    image = tf.image.central_crop(image, central_fraction=central_fraction)

  if out_height and out_width:
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_image_with_pad(image, out_height, out_width)
    image = tf.squeeze(image, [0])

  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image

def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
            Lenght of this list could be less than batch_size, in this case only
            first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            img_array = imread(f, mode='RGB').astype(np.float32)
            #img_tensor = tf.convert_to_tensor(img_array, np.float32)
            image = preprocess_for_eval(img_array)
            #image = imresize(imread(f, mode='RGB'), [FLAGS.image_height, FLAGS.image_width]).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def graph(x, y, i, x_max, x_min, grad, aug_x=None):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001

    if USE_AUTO_AUG:
        aug_x = autoaug_diversity(x)    # x -> [aug_type*bs, w, h, c]
        y = tf.tile(y, [AUG_num, 1])    # y -> [aug_type*bs, 1]

        logits_weights = AUG_weights_1
        aux_logits_weights = AUG_weights_04

    elif FLAGS.use_si:
        aug_x = si_diversity(x)         # x -> [5*bs, w, h, c]
        y = tf.tile(y, [5, 1])          # y -> [5*bs, 1]

        logits_weights = SI_weights
        aux_logits_weights = SI_weights_04

    else:
        aug_x = x

        logits_weights = 1.0
        aux_logits_weights = 0.4



    # should keep original x here for output
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(aug_x), num_classes=num_classes, is_training=False, reuse=True)

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(aug_x), num_classes=num_classes, is_training=False, reuse=True)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(aug_x), num_classes=num_classes, is_training=False, reuse=True)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
            input_diversity(aug_x), num_classes=num_classes, is_training=False, reuse=True)

    if FLAGS.target_model == 'ens':
        logits = (logits_v3 + logits_v4 + logits_res_v2 + logits_resnet) / 4
        auxlogits = (end_points_v3['AuxLogits'] + end_points_v4['AuxLogits'] + end_points_res_v2['AuxLogits']) / 3

    elif FLAGS.target_model == 'resnet':
        logits = logits_resnet

    elif FLAGS.target_model == 'inception_v3':
        logits = logits_v3
        auxlogits = end_points_v3['AuxLogits']

    elif FLAGS.target_model == 'inception_v4':
        logits = logits_v4
        auxlogits = end_points_v4['AuxLogits']

    elif FLAGS.target_model == 'inception_resnet_v2':
        logits = logits_res_v2
        auxlogits = end_points_res_v2['AuxLogits']

    else:
        assert False, "Unknown arch."


    cross_entropy = tf.losses.softmax_cross_entropy(y,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=logits_weights)
    if FLAGS.target_model != 'resnet':
        cross_entropy += tf.losses.softmax_cross_entropy(y,
                                                         auxlogits,
                                                         label_smoothing=0.0,
                                                         weights=aux_logits_weights)

    noise = tf.gradients(cross_entropy, x)[0]
    if FLAGS.use_ti:
        noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    if USE_AUTO_AUG:
        y = tf.reshape(y, [AUG_num, FLAGS.batch_size, -1])
        y = y[0]
    elif FLAGS.use_si:
        y = tf.reshape(y, [5, FLAGS.batch_size, -1])
        y = y[0]
    else:
        pass

    if DEBUG_:
        return x, y, i, x_max, x_min, noise, aug_x
    else:
        return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad, aug_x=None):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def input_diversity(input_tensor):

    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)




def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)


    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        '''
        # old version
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)
        y = tf.one_hot(predicted_labels, num_classes)
        '''

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_v4, end_points_v4 = inception_v4.inception_v4(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
                x_input, num_classes=num_classes, is_training=False)

        if FLAGS.target_model == 'ens':
            predicted_labels = tf.argmax(end_points_resnet['predictions']+end_points_v3['Predictions']+end_points_v4['Predictions']+end_points_res_v2['Predictions'], 1)

        elif FLAGS.target_model == 'resnet':
            predicted_labels = tf.argmax(end_points_resnet['predictions'], 1)

        elif FLAGS.target_model == 'inception_v3':
            predicted_labels = tf.argmax(end_points_v3['Predictions'], 1)

        elif FLAGS.target_model == 'inception_v4':
            predicted_labels = tf.argmax(end_points_v4['Predictions'], 1)

        elif FLAGS.target_model == 'inception_resnet_v2':
            predicted_labels = tf.argmax(end_points_res_v2['Predictions'], 1)

        else:
            assert False, "Unknown arch."

        y = tf.one_hot(predicted_labels, num_classes)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)

        if DEBUG_:
            x_aug = tf.zeros(shape=[AUG_num*FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])
            x_adv, _, _, _, _, _, x_aug = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad, x_aug])
        else:
            x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])



        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        with tf.Session() as sess:
            s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
            s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
            s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
            s8.restore(sess, FLAGS.checkpoint_path_resnet)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                if DEBUG_:
                    out = sess.run(x_aug, feed_dict={x_input: images})
                    pdb.set_trace()

                adv_images = sess.run(x_input, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)



if __name__ == '__main__':
    tf.app.run()
