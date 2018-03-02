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
    def __init__(self, class_name, sess, inference_model, feed_dict, image, label, cell_width, cell_height, ratio, probability_threshold, iou_threshold):
        self.sess = sess
        self.yolo = inference_model
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.ratio = ratio
        self.prob_th = probability_threshold
        self.iou_th = iou_threshold
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
            # [num cells, bounding box, (x,y)]
            # self.yolo.score: pr(class) * iou, [num cells, bounding box, num classs]
        image_xy_min, image_xy_max, score = self.sess.run([self.yolo.cell_center_xy_min[0], self.yolo.cell_center_xy_max[0], self.yolo.scores[0]], feed_dict=self.feed_dict) 
        boxes = self.non_max_suppression(image_xy_min, image_xy_max, score, self.prob_th, self.iou_th)
        object_count = 0

        # xy_min, xy_max scaled to cell index
        for _xy_min, _xy_max, _score in boxes:
            largest_index = np.argmax(_score)
            if _score[largest_index] > probability_threshold:
                width_height = _xy_max - _xy_min
                xy_min_scaled = _xy_min * self.ratio
                width_height_scaled = width_height * self.ratio

                object_rectangle = pathes.Rectangle(xy_min_scaled, width_height_scaled[0], width_height_scaled[1], facecolor='none', linewidth=3, edgecolor=self.colors[largest_index])
                self.ax.add_patch(object_rectangle)
                self.ax.annotate(self.class_name[largest_index] + '%.3f%%' % (_score[largest_index]*100), xy_min_scaled) 
                object_count += 1
        
        self.fig.canvas.set_window_title('%d objects' % object_count)

    # Choose one bounding box
    def non_max_suppression(self, image_xy_min, image_xy_max, confidence_score):
        # [num cells*bounding boxes, None]
        _image_xy_min = np.reshape(image_xy_min, [-1,2])
        _image_xy_max = np.reshape(image_xy_max, [-1,2])
        _confidence_score = np.reshape(confidence_score, [-1, len(self.class_name)])

        box_info = [(_xy_min, _xy_max, _score) for _xy_min, _xy_max, _score in zip(_image_xy_min, _image_xy_max, _confidence_score)]
        for class_index in range(len(self.class_name)):
            # sort box_info in largest probability order
                # reverse=True -> descending order
            box_info.sort(key=lambda box: box[2][class_index], reverse=True)
            
            for i in range(len(box_info)-1):
                # If conditional class probability is lower than threshold, do not care that bounding box
                if box_info[i][2][class_index] < self.prob_th:
                    continue
                
                # Non_max_suppression, kill other candidates
                for j in box_info[i+1:]:
                    if self.iou(box_info[i][0], box_info[i][1], j[0], j[1]) > self.iou_th:
                        j[2][class_index] = 0

        return box_info


    def iou(box1_xy_min, box1_xy_max, box2_xy_min, box2_xy_max):
        box1_area = np.multiply.reduce(box1_xy_max - box1_xy_min, -1)
        box2_area = np.multiply.reduce(box2_xy_max - box2_xy_max, -1)

        # Choose element-wise maximum and minimum
        overlapped_min = np.maximum(box1_xy_min, box2_xy_min)
        overlapped_max = np.minimum(box1_xy_max, box2_xy_max)
        overlapped_area = np.multiply.reduce(overlapped_max - overlapped_min, -1)

        total_area = box1_area + box2_area - overlapped_area
        iou = overlapped_area / np.maximum(total_area, 1e-8)
        return iou



