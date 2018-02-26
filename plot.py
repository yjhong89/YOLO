import tensorflow as tf
import numpy as np
import utils
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as patches
# Make iterable object
import itertools


class plot():
    def __init__(self, class_name, sess, inference_model, feed_dict, image, label, cell_width, cell_height):
        self.sess = sess
        self.yolo = inference_model
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.feed_dict = feed_dict
        self.image = image
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
        self.ax.set_xticks(np.arange(0, self.image_width, self.image_width / self.cell_width))
        self.ax.set_yticks(np.arange(0, self.image_height, self.image_height / self.cell_height))
        # grid: 'which': major and minor tick grids
        self.ax.grid(which='both', axis='both')

    def plot_real(self):
        self.ax.imshow(self.image / 255.0)
        # self.labels: real labels, (object_appear, object_relative_xy, class_prob, regression_coord_label)
            # [batch_size, num_cells, 1], [batch_size, num_cells, 1, 4], [batch_size, num_cells, num_class] [batch_size, num_cells, 1, 4]
        # Select object appear cell
        for cell_index, (_object_appear, _object_relative_xy, _class_prob, _regression_coord_label) in enumerate(zip(self.label[0], self.label[1], self.label[2], self.label[3])):
            if _object_appear != 0:
                most_prob_index = np.argmax(_class_prob)
                # Getting index
                cell_height_index = cell_index // self.cell_width
                cell_width_index = cell_index % self.cell_height
                # Bottom and left coordinate
                #cell_rectangle = patches.Rectangle((cell_width_index * self.image_width / self.cell_width, cell_height_index * self.image_height / self.cell_height), self.image_width / self.cell_width, self.image_height / self.cell_height, linestyle='dashed', edgecolor=self.colors[most_prob_index])
                #self.real_plots.append(self.ax.add_patch(cell_rectangle))
    
                # All between [0,1]
                offset_x, offset_y, w_sqrt, h_sqrt = _regression_coord_label[0]
                cell_x = cell_width_index + offset_x
                cell_y = cell_height_index + offset_y
                # Note that width and height are normalized to whole image
                w, h = w_sqrt*w_sqrt, h_sqrt*h_sqrt             
                w_real, h_real = w * self.image_width, h * self.image_height
                x_min, y_min = cell_x * self.image_width / self.cell_width - w_real / 2, cell_y * self.image_height / self.cell_height - h_real / 2
                object_rectangle = patches.Rectangle((x_min, y_min), w_real, h_real, facecolor='none', linewidth = 3, edgecolor=self.colors[most_prob_index])
                self.real_plots.append(self.ax.add_patch(object_rectangle))
                # ax.annotation(string, (x,y) to annotate)
                object_annotation = self.ax.annotate(self.class_name[most_prob_index], (x_min, y_min))
                self.real_plots.append(object_annotation)
                print(self.class_name[most_prob_index])
                print(cell_x * self.image_width / self.cell_width, cell_y * self.image_height / self.cell_height)
           
        return self.real_plots 
        

    def plot_pred(self):
        # self.yolo's attributes has batch size axis
            # Per each bounidng box, represents coordinate and confidence score
        image_xy_min, image_xy_max, score = self.sess.run([self.yolo.cell_center_xy_min[0], self.yolo.cell_center_xy_max[0], self.yolo.scores[0]], feed_dict=self.feed_dict) 
