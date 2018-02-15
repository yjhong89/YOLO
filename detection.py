import tensorflow as tf
import numpy as np
import configparser
import utils
import os
import tensorflow.contrib.slim as slim
from model import yolo_model
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as patches
# Make iterable object
import itertools

class plot():
    def __init__(self, class_name, sess, inference_model, feed_dict, image, label):
        self.sess = sess
        self.yolo = inference_model
        # image, label include batch size axis
        self.image = np.squeeze(image, axis=0)
        self.label = label
        for i, j  in enumerate(self.label):
             self.label[i] = np.squeeze(j, axis=0)
        self.class_name = class_name
        self.real_plots = list()
        self.fig = plt.figure()
        # Get Current axes, creatinig one if needed
        self.ax = self.fig.add_subplot(111, aspect='equal')        
        # itertools.cycle: Repeatedely produce
        # plt.rcParams['axes.color_cycle'] returns list of 10 colors
        # plt.rcParams['axes.prop_cycle'] returns cycler('color', 10 kinds of colors)
            # -> Need to slice with 'zip'
            # hamait.tistory.com/803
        self.colors = [color['color'] for cl_name, color in zip(class_name, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
        self.image_height, self.image_width, _ = self.image.shape
        self.draw()
        self.plot_real()
        self.fig.savefig('./test.pdf', dpi=120) 
        for i in self.real_plots:
            i.remove()


    def draw(self):
        # Set x-axis tick (448,448) divide to 7*7 grids
        self.ax.set_xticks(np.arange(0, self.image_width, self.image_width / self.yolo.cell_width))
        self.ax.set_yticks(np.arange(0, self.image_height, self.image_height / self.yolo.cell_height))
        # grid: 'which': major and minor tick grids
        self.ax.grid(which='both', axis='both')

    def plot_real(self):
        self.ax.imshow(self.image)
        # self.labels: real labels, (object_appear, object_relative_xy, class_prob, regression_coord_label)
            # [batch_size, num_cells, 1], [batch_size, num_cells, 1, 4], [batch_size, num_cells, num_class] [batch_size, num_cells, 1, 4]
        # Select object appear cell
        for cell_index, (_object_appear, _object_relative_xy, _class_prob, _regression_coord_label) in enumerate(zip(self.label[0], self.label[1], self.label[2], self.label[3])):
            if _object_appear != 0:
                most_prob_index = np.argmax(_class_prob)
                # Getting index
                cell_height_index = cell_index // self.yolo.cell_width
                cell_width_index = cell_index % self.yolo.cell_height
                # Bottom and left coordinate
                cell_rectangle = patches.Rectangle((cell_width_index * self.image_width / self.yolo.cell_width, cell_height_index * self.image_height / self.yolo.cell_height), self.image_width / self.yolo.cell_width, self.image_height / self.yolo.cell_height, linestyle='dashed', edgecolor=self.colors[most_prob_index])
                self.real_plots.append(self.ax.add_patch(cell_rectangle))
    
                # All between [0,1]
                offset_x, offset_y, w_sqrt, h_sqrt = _regression_coord_label[0]
                cell_x = cell_width_index + offset_x
                cell_y = cell_height_index + offset_y
                # Note that width and height are normalized to whole image
                w, h = w_sqrt*w_sqrt, h_sqrt*h_sqrt             
                w_real, h_real = w * self.image_width, h * self.image_height
                x_min, y_min = cell_x * self.image_width / self.yolo.cell_width - w_real / 2, cell_y * self.image_height / self.yolo.cell_height - h_real / 2
                object_rectangle = patches.Rectangle((x_min, y_min), w_real, h_real, linewidth = 1, color=self.colors[most_prob_index])
                self.real_plots.append(self.ax.add_patch(object_rectangle))
                # ax.annotation(string, (x,y) to annotate)
                object_annotation = self.ax.annotate(self.class_name[most_prob_index], (x_min, cell_y * self.image_height /self.yolo.cell_height + h_real /2))
                self.real_plots.append(object_annotation)
                print(self.class_name[most_prob_index])
                print(cell_x * self.image_width / self.yolo.cell_width, cell_y * self.image_height / self.yolo.cell_height)
           
        return self.real_plots 
        

    def plot_pred(self):
        # self.yolo's attributes has batch size axis
        x_center, y_center, w, h, iou, prob = self.sess.run([self.yolo.coord[0][:,:,0], self.yolo.coord[0][:,:,1], self.yolo.coord[0][:,:,2], self.yolo.coord[0]]:,:,3], self.yolo.iou[0], self.yolo.class_prob_pred[0]) 
        


def detect(config, args):
    base_dir = os.path.expanduser(config.get('config', 'basedir'))
    log_dir = os.path.join(base_dir, config.get('config', 'logdir'))
    cache_dir = os.path.join(base_dir, config.get('cache', 'cachedir'))

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
        images, labels = utils.read_tfrecord(data_path, config, num_class)
        images_normalized = tf.image.per_image_standardization(images)

        sample_detect = tf.train.shuffle_batch((images,) + labels, batch_size=1, capacity=100, min_after_dequeue=50, num_threads=1)

        detect_image = tf.placeholder(tf.float32, [1] + images.get_shape().as_list(), name='detect_image')
        detect_label = [tf.placeholder(label.dtype, [1] + label.get_shape().as_list(), name='detect_label' + label.op.name) for label in labels]
        #global_step = tf.train.get_or_create_global_step()
        yolo = yolo_model(config, args, num_class)
        # Do inference
        yolo(detect_image, is_training=False)

        with tf.name_scope('objective'):
            yolo.create_objective(*sample_detect[1:])
    
        # Can partially restore variable with 'exclude'
        variables_to_restore = slim.get_variables_to_restore()

        # Must be located here, before session run
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # Get data by running session
        _image = sess.run(sample_detect[0])
        plt.imsave('check.png',np.squeeze(_image, axis=0))
        _labels = sess.run(sample_detect[1:])

        # This way gets always same data
        #_image, _labels, _image_normalized = sess.run([images, labels, images_normalized])
        #plt.imsave('check.png', _image)
        
        # expand batch size dimension for real label
        feed_dict = dict([(label_placeholder, real_labels) for label_placeholder, real_labels in zip(detect_label, _labels)])
        feed_dict[detect_image] = utils.image_per_standardization(_image)

        model_path = tf.train.latest_checkpoint(log_dir)

        # Restore
        slim.assign_from_checkpoint_fn(model_path, variables_to_restore)(sess)

        coord.request_stop()
        coord.join(threads)
        
        # Plot
        _ = plot(class_names, sess, yolo, feed_dict, _image, _labels)
