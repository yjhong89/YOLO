import sys
sys.path.append('..')
import tensorflow as tf
import utils
import importlib
import numpy as np

class yolo_model():
    def __init__(self, config, args, num_class):
        self.config = config
        self.args = args
        # yolo is 'net/yolo.py'
        # yolo2 is 'net/yolo2.py'
        self.yolo_name = self.config.get('config', 'model')
        yolo_module = importlib.import_module(self.yolo_name + '.' + self.yolo_name)
        print(yolo_module.__name__)

        self.num_classes = num_class   
        self.boxes_per_cell = self.config.getint(self.yolo_name, 'boxes_per_cell')
        self.channel = self.config.getint(self.yolo_name, 'channel')
        self.cell_width = self.config.getint(self.yolo_name, 'width') // self.config.getint(self.yolo_name, 'ratio')
        self.cell_height = self.config.getint(self.yolo_name, 'height') // self.config.getint(self.yolo_name, 'ratio')
       
        # Call yolo network function 
        self.yolo = getattr(yolo_module, self.yolo_name)

    def __call__(self, image, is_training=True):
        num_cells = self.cell_width * self.cell_height
        results = self.yolo(net = image, is_training=is_training, classes=self.num_classes, boxes_per_cell=self.boxes_per_cell, channel=self.channel) 
        # self.results: [batch size, 7*7*30], S*S*(B*5+C)
        with tf.name_scope('regression_input'):
            x_y_w_h_conf = tf.reshape(results[:, :num_cells*self.boxes_per_cell*5], [-1, num_cells, self.boxes_per_cell, 5], name='x_y_w_h_conf')
            self.class_prob_pred = tf.reshape(results[: ,num_cells*self.boxes_per_cell*5:], [-1, num_cells, self.num_classes], name='class_probabilities')
            self.conf = x_y_w_h_conf[:,:,:,0]    
            self.x_y = x_y_w_h_conf[:,:,:,1:3]
            # width and height regression is based on square root, so assume network outputs square root of width and height
            # Note that it should be positive
            w_h = tf.abs(x_y_w_h_conf[:,:,:,3:], name='w_h')
            self.coord = tf.concat([self.x_y, w_h], axis=-1, name='coordinate')
            # Get width and height and to calculate area, componsate cell grid index
            area_w_h = tf.square(x_y_w_h_conf[:,:,:,3:]) * np.array([self.cell_width, self.cell_height])
            self.area_xy_min = self.x_y - area_w_h / 2
            self.area_xy_max = self.x_y + area_w_h / 2
            # Shape of [batch_size, num_cell ,bounding_box], reduce last axis
            self.predicted_area = tf.reduce_prod(area_w_h, -1, name='predicted_area')
        

    def create_objectives(self, object_appear, object_relative_xy, class_prob, regression_coord_label):
        '''
            object_appear: [num_cells, 1]
            object_relative_xy: [num_cells, 1, 4] (x_min, y_min, x_max, y_max)
            class_prob: [num_cell, num_class]
            regression_coord_label: [num_cell, 1, 4] (x,y,w,h)
            Including batch size axis at dimension 0
        '''
        # Sum square error because it is easy to optimize
        with tf.name_scope('iou'):
            # xy_max - xy_min: w,h
            target_area = tf.reduce_prod(object_relative_xy[:,:,:,2:] - object_relative_xy[:,:,:,:2], -1, name='target_area')
            # xy_min for overlapped box, element-wise
            overlapped_xy_min = tf.maximum(self.area_xy_min, object_relative_xy[:,:,:,:2])
            # xy_max for overlapped box, element-wise
            overlapped_xy_max = tf.minimum(self.area_xy_max, object_relative_xy[:,:,:,2:])
            overlapped_area = tf.reduce_prod(overlapped_xy_max - overlapped_xy_min, -1, name='overlapped_area')
            total_area = tf.maximum(self.predicted_area + target_area - overlapped_area, 1e-5, name='total_area')
            self.iou  = tf.truediv(overlapped_area, total_area, name='IOU')
        # Want one bounding box predictor to be responsible for each object.
        # We assign one predictor to be responsible for predicting an object based on which prediction has the highest current iou
            # In equation (3), 1_ij denotes that the j th bounding box predictor in cell i is responsible for prediction
        with tf.name_scope('responsible'):
            # self.iou: [batch_size, num_cell, bounding_box]
                # -> [batch_size, num_cell, 1]
            highest_iou = tf.reduce_max(self.iou, 2, keep_dims=True, name='highest_iou')
            # Filtering highest_iou
                # tf.equal([batch_size, num_cell, bounding_box], [batch_size, num_cell, 1]) -> [batch_size, num_cell, bounding_box]
            highest_iou_index = tf.to_float(tf.equal(self.iou, highest_iou), name='highest_iou_index')
            # object_appear: [batch_size, num_cell, 1]
                # [batch_size, num_cell, bounding_box]
            self.only_highest_iou = highest_iou_index * object_appear
            # For no_obj
            self.no_obj_iou = 1 - self.only_highest_iou
        
        # Objectives have 4 dimensions, we need mask dimension to be expanded 
        with tf.name_scope('regression_losses'):
            coordinate_objective = self.config.getfloat(self.yolo_name, 'coord') * tf.reduce_sum(tf.expand_dims(self.only_highest_iou, -1) * tf.square(self.coord - regression_coord_label), name='coordinate_regresssion')
            #print(self.only_highest_iou.get_shape().as_list())
            tf.summary.scalar('coordinate_objective', coordinate_objective)
            confidence_obj = tf.reduce_sum(self.only_highest_iou * tf.square(self.conf - self.only_highest_iou), name='confidence_obj_regression')
            confidence_noobj = self.config.getfloat(self.yolo_name, 'noobj') * tf.reduce_sum(self.no_obj_iou * tf.square(self.conf - self.only_highest_iou), name='confidence_noobj_regression')
            tf.summary.scalar('confidence_objective', confidence_obj + confidence_noobj)
            probability_objective = tf.reduce_sum(object_appear * tf.square(class_prob - self.class_prob_pred), name='probability_regression')
            tf.summary.scalar('probability_objective', probability_objective)

            total_loss = coordinate_objective + confidence_obj + confidence_noobj + probability_objective
            tf.summary.scalar('total_loss_sum', total_loss)
            # If we use 'tf.losses', any loss is added to teh tf.GraphKeys.LOSSES collection,
            # We can call them easily with tf.losses.get_total_loss()
            tf.add_to_collection(tf.GraphKeys.LOSSES, total_loss)


