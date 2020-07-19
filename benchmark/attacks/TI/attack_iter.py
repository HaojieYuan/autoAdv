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

tf.flags.DEFINE_float('prob', 0.4, 'probability of using diverse inputs.')

tf.flags.DEFINE_string('autoaug_file', '', 'auto augmentation search result.')



FLAGS = tf.flags.FLAGS

if FLAGS.prob !=0 :
    print("Using DI.")

if FLAGS.use_ti:
    print("Using TI.")

if os.path.exists(FLAGS.autoaug_file):
    print("Using auto augment.")
    USE_AUTO_AUG = True
    with open(FLAGS.autoaug_file) as f:
        AUG_POLICY = eval(f.readline())
    AUG_weights = [aug_weight for aug_type, aug_weight, aug_prob, aug_range in AUG_POLICY]
    w_sum = sum(AUG_weights)
    AUG_weights = [aug_weight/w_sum for aug_weight in AUG_weights]

    AUG_weights_1 = FLAGS.batch_size*AUG_weights
    AUG_weights_04 = [weight_*0.4 for weight_ in AUG_weights_1]

    AUG_num = len(AUG_POLICY)
else:
    USE_AUTO_AUG = False

AUG_TYPE = {0: 'resize_padding', 1: 'translation', 2: 'rotation',
            3: 'gaussian_noise', 4: 'horizontal_flip', 5: 'vertical_flip'}


def augmentation(type, prob, mag_range, input_tensor):

    op_type = AUG_TYPE[type]
    mag_range = int(mag_range)

    if op_type == 'resize_padding':
        mag = tf.random_uniform((), 0, mag_range, dtype=tf.int32)
        mag = tf.cast(mag, tf.float32)
        w_modified = 2*int(0.01*mag*FLAGS.image_width)
        h_modified = 2*int(0.01*mag*FLAGS.image_width)
        w_resized = FLAGS.image_width - w_modified
        h_resized = FLAGS.image_width - h_modified

        rescaled = tf.image.resize_images(input_tensor, [w_resized, h_resize],
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



def autoaug_diversity(input_tensor):
    auged_list = [augmentation(aug_type, aug_prob, aug_range, input_tensor) for \
                               aug_type, aug_weight, aug_prob, aug_range in AUG_POLICY]

    return tf.concat(auged_list, 0) # concat on 0 axis



def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

kernel = gkern(15, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)

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
            image = imresize(imread(f, mode='RGB'), [FLAGS.image_height, FLAGS.image_width]).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
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

    else:
        aug_x = x



    # should keep original x here for output
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(aug_x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(aug_x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(aug_x), num_classes=num_classes, is_training=False, reuse=True)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
            input_diversity(aug_x), num_classes=num_classes, is_training=False)


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


    if not USE_AUTO_AUG:
        cross_entropy = tf.losses.softmax_cross_entropy(y,
                                                        logits,
                                                        label_smoothing=0.0,
                                                        weights=1.0)
        if FLAGS.target_model != 'resnet':
            cross_entropy += tf.losses.softmax_cross_entropy(y,
                                                             auxlogits,
                                                             label_smoothing=0.0,
                                                             weights=0.4)
    else:
        #logits = tf.reshape(logits, [AUG_num, FLAGS.batch_size, -1])
        #if FLAGS.target_model != 'resnet':
        #    auxlogits = tf.reshape(auxlogits, [AUG_num, FLAGS.batch_size, -1])
        #y = tf.reshape(y, [AUG_num, FLAGS.batch_size, -1])

        cross_entropy = tf.losses.softmax_cross_entropy(y,
                                                        logits,
                                                        label_smoothing=0.0,
                                                        weights=AUG_weights_1)
        if FLAGS.target_model != 'resnet':
                cross_entropy += tf.losses.softmax_cross_entropy(y,
                                                                 auxlogits,
                                                                 label_smoothing=0.0,
                                                                 weights=AUG_weights_04)
        '''
        cross_entropy = 0

        for i in range(AUG_num):
            cross_entropy += AUG_weights[i]*tf.losses.softmax_cross_entropy(y[i],
                                                                         logits[i],
                                                                         label_smoothing=0.0,
                                                                         weights=1.0)
            if FLAGS.target_model != 'resnet':
                cross_entropy += AUG_weights[i]*tf.losses.softmax_cross_entropy(y[i],
                                                                             auxlogits[i],
                                                                             label_smoothing=0.0,
                                                                             weights=0.4)
        '''


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

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)
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

                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)



if __name__ == '__main__':
    tf.app.run()