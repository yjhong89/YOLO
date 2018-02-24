import tensorflow as tf
import os
import utils
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import pandas as pd
import shutil
import importlib
import sys
sys.path.append('./')

# Learning rate will be adjusted
def get_optimizer(config, name):
    section_name = 'optimizer'
    learning_rate = config.getfloat(section_name, 'learning_rate')
    optimizer_dict = {'adam': tf.train.AdamOptimizer(learning_rate, config.getfloat(section_name, name+'beta1'), config.getfloat(section_name, name+'beta2'), config.getfloat(section_name, name+'epsilon')),
                    'momentum': tf.train.MomentumOptimizer(learning_rate, config.getfloat(section_name, 'momentum'))
                    }

    return optimizer_dict[name]

def _get_vars_and_update_ops(scope):
    is_trainable = lambda x : x in tf.trainable_variables()
    scope_var_list = list(filter(is_trainable, slim.get_model_variables(scope)))
    # For batch normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

    tf.logging.info('trainable variables')
    for i in scope_var_list:
        print(i.op.name)
    tf.logging.info('update ops')
    for i in update_ops:
        print(i.op.name)

    return scope_var_list, update_ops

def train(config, args, anchor_info, model_name):
    base_dir = os.path.expanduser(config.get('config', 'basedir'))
    log_dir = os.path.join(os.path.join(base_dir, config.get('config', 'logdir')), model_name)
    cache_dir = os.path.join(base_dir, config.get('cache', 'cachedir'))


    class_txt = os.path.join(base_dir, config.get('cache', 'name'))
    with open(class_txt, 'r') as f:
        # .strip() removes '\n'
        class_names = [line.strip() for line in f]
    num_class = len(class_names)
    tf.logging.info('%d classes' % num_class)

    os.makedirs(log_dir, exist_ok=True)
    if args.delete:
        shutil.rmtree(log_dir)

    cell_width = config.getint(model_name, 'width') // config.getint(model_name, 'ratio')
    cell_height = config.getint(model_name, 'height') // config.getint(model_name, 'ratio')
    
    yolo_path = importlib.import_module(model_name + '.model')
    # Call class module
    yolo_model = getattr(yolo_path, 'yolo_model')

    if model_name == 'yolo':
        yolo = yolo_model(config, args, num_class)
    elif model_name == 'yolo2':
        yolo = yolo_model(config, args, anchor_info, num_class)
    else:
        raise ValueError('Not supproted yolo model')


    # train.tfrecord, val.tfrecord
    data_path = [os.path.join(cache_dir, types) + '.tfrecord' for types in args.data_type]
    # Calculate number of data
    num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in data_path)
    tf.logging.info('%d examples' % num_examples)

    with tf.name_scope('train_batch'):
        # labels = (object_appear, object_relative_xy, class_prob, regression_coord_label) (tuple)
        images, labels, image_path = utils.read_tfrecord(data_path, config, num_class, cell_width, cell_height)
        # Image normalization (x - mean) / stddev, input_image_shape: [height, width, channels]
        images = tf.image.per_image_standardization(images)
        # When add tuples, need 'comma' for single element
        batch = tf.train.shuffle_batch((images,) + labels, batch_size=args.batch_size, capacity=args.batch_size*50, min_after_dequeue=args.batch_size*10, num_threads=10)
     

    global_step = tf.train.get_or_create_global_step()
    # Return list, [image, label[0], label[1]....]
    # __call__ method 
    yolo(batch[0], is_training=True)

    vars_update_ops = _get_vars_and_update_ops(yolo.yolo_name)

    with tf.name_scope('total_loss'):
        # Passing label list argument
        yolo.create_objectives(*batch[1:])
        weight_decay_loss = config.getfloat('optimizer', 'weight_decay') * tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in vars_update_ops[0]]))
        tf.add_to_collection(tf.GraphKeys.LOSSES, weight_decay_loss) 
        total_loss = tf.losses.get_total_loss('total_loss')

    with tf.name_scope('optimizer'):
        decay_steps = config.getint('exponential_decay', 'decay_steps')
        decay_rate = config.getfloat('exponential_decay', 'decay_rate')
        learning_rate = tf.train.exponential_decay(config.getfloat('optimizer', 'learning_rate'), global_step, decay_steps, decay_rate, staircase=True)
        optimizer = get_optimizer(config, args.optimizer_name)
        tf.logging.info('%s optimizer' % (args.optimizer_name))


    train_op = slim.learning.create_train_op(total_loss, optimizer, update_ops=vars_update_ops[1], variables_to_train=vars_update_ops[0], clip_gradient_norm=args.gradient_norm, summarize_gradients=False)

    init_fn = lambda sess : tf.logging.info('%d global steps' % sess.run(global_step))
   
    # Check slim.learning.train 
    slim.learning.train(train_op, log_dir, master='', is_chief=(args.task == 0), global_step=global_step, number_of_steps=args.steps, init_fn=init_fn, summary_writer=tf.summary.FileWriter(log_dir), save_summaries_secs=args.summary_secs, save_interval_secs=args.save_secs)

