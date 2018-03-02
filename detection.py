import tensorflow as tf
import numpy as np
import configparser
import utils
from plot import plot
import os
import tensorflow.contrib.slim as slim
import importlib


def detect(config, args, anchor_info, model_name):
    base_dir = os.path.expanduser(config.get('config', 'basedir'))
    log_dir = os.path.join(os.path.join(base_dir, config.get('config', 'logdir')), model_name)
    cache_dir = os.path.join(base_dir, config.get('cache', 'cachedir'))

    yolo_model = getattr(importlib.import_module(model_name+'.model'), 'yolo_model')
    ratio = config.getint(model_name, 'ratio')
    cell_height = config.getint(model_name, 'height') // ratio
    cell_width = config.getint(model_name, 'width') // ratio

    class_txt = os.path.join(base_dir, config.get('cache', 'name'))
    with open(class_txt, 'r') as f:
        class_names = [line.strip() for line in f]
    num_class = len(class_names)

    data_path = [os.path.join(cache_dir, 'val') + '.tfrecord']

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())

        # labels = {object_appear, object_relative_xy, class_prob, regression_coord_label)
        images, labels, image_path = utils.read_tfrecord(data_path, config, num_class, cell_width, cell_height)
        images_normalized = tf.image.per_image_standardization(images)

        sample_detect = tf.train.shuffle_batch((images,) + labels, batch_size=1, capacity=100, min_after_dequeue=50, num_threads=10)

        detect_image = tf.placeholder(tf.float32, [1] + images.get_shape().as_list(), name='detect_image')
        detect_label = [tf.placeholder(label.dtype, [1] + label.get_shape().as_list(), name='detect_label' + label.op.name) for label in labels]

        global_step = tf.train.get_or_create_global_step()
        
        if model_name == 'yolo':
            yolo = yolo_model(config, args, num_class)
        elif model_name == 'yolo2':
            yolo = yolo_model(config, args, anchor_info, num_class)
        else:
            raise ValueError('Not supported yolo')

        # Do inference
        yolo(detect_image, is_training=False)

        with tf.name_scope('objective'):
            yolo.create_objectives(*sample_detect[1:])
    
        # Can partially restore variable with 'exclude'
        variables_to_restore = slim.get_variables_to_restore()

        # Must be located here, before session run
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)


        # Do not call session run twice -> different label result to mixing labels
        _image, _labels = sess.run([sample_detect[0], sample_detect[1:]])
        plt.imsave('check5.jpg',sess.run(images)/255.0)
        #_labels = sess.run(sample_detect[1:])

        # This way gets always same data
        #_image, _labels, _image_normalized = sess.run([images, labels, images_normalized])
        #plt.imsave('check.png', _image)
        
        # expand batch size dimension for real label
        feed_dict = dict([(label_placeholder, real_labels) for label_placeholder, real_labels in zip(detect_label, _labels)])
        feed_dict[detect_image] = utils.image_per_standardization(_image)

        model_path = tf.train.latest_checkpoint(log_dir)

        # Restore
        #slim.assign_from_checkpoint_fn(model_path, variables_to_restore)(sess)

        coord.request_stop()
        coord.join(threads)
        
        # Plot
        _ = plot(class_names, sess, yolo, feed_dict, _image, _labels, cell_width, cell_height, ratio, args.probability_threshold, args.iou_threshold)
