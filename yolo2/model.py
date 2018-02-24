import tensorflow as tf
import importlib
import os
import numpy as np
import pandas as pd

class yolo_model():
    def __init__(self, config, args, anchors, num_classes):
        self.config = config
        self.args = args
        self.num_classes = num_classes
        # 5,2 array(width, height)
        self.anchors = anchors
       
        # yolo2
        self.yolo_name = self.config.get('config', 'model')
        yolo_module = importlib.import_module(self.yolo_name + '.' + self.yolo_name)
        self.channel = self.config.getint(self.yolo_name, 'channel')
        self.coord_hparam = self.config.getfloat(self.yolo_name, 'coord')
        self.noobj_hparam = self.config.getfloat(self.yolo_name, 'noobj')

        # Call yolo networks
        self.yolo = getattr(yolo_module, self.yolo_name)
        

    def __call__(self, image, is_training=True):
        '''
            Direct location prediction
            Most instability comes from predicting the (x,y) locations for the box
        '''
        # We get an output feature map of 13*13*125
        results = self.yolo(net=image, is_training=is_training, num_anchors=len(self.anchors), classes=self.num_classes, channel=self.channel)
        # 13, 13
        _, ofm_x, ofm_y, _ = results.get_shape().as_list()
        num_cells = ofm_x * ofm_y
        with tf.name_scope('regression_input'):
            # Predict 5 boxes with 5 coordinates each and 20 classes per box(anchor)
            # 5*(5+20) = 125
            # t_x,t_y,t_o, t_w,t_h
            result_reshaped = tf.reshape(results, [-1, num_cells, len(self.anchors), 5+self.num_classes], name='result_reshape')
            '''
                Instead of predicting offsets, predict location coordinates as yolo
                Use a logistic activation to constrain the noetwork's prediction to fall between [0,1]-> t_x, t_y, t_o
            '''
            logistic_activation = tf.nn.sigmoid(result_reshaped[:,:,:,:3])
            # t_x,t_y
            self.center_xy = logistic_activation[:,:,:,:2]
            # t_o
            self.t_o = logistic_activation[:,:,:,2]
            # class probaility, [batch size, num cell, bounding box, class_prob(20)]
            self.class_prob = result_reshaped[:,:,:,5:]
            # With bounding box prior p_w, p_h, predict b_w, b_h
                # (5,2) -> (1,1,5,2)
            self.wh = tf.exp(result_reshaped[:,:,:,3:5]) * np.reshape(self.anchors, [1,1,len(self.anchors), -1])

            # Figure3, predict the width and height of the box as offsets from cluster centroids
            self.area = tf.reduce_prod(self.wh, axis=[-1], name='predicted_area')
            self.area_xy_min = self.center_xy - self.wh / 2
            self.area_xy_max = self.center_xy + self.wh / 2
            # Make width and height relative to the whole output feature, to do regression
            self.relative_wh_sqrt = tf.sqrt(self.wh / np.array([ofm_x, ofm_y]), name='wh_sqrt')
            self.coord = tf.concat([self.center_xy, self.relative_wh_sqrt], axis=-1, name='coordination')           
            

    # Same regression problem as yolo
    def create_objectives(self, object_appear, object_relative_xy, class_prob, regression_coord_label):
        tf.logging.info('Creating objective functions')
        '''
            object_appear: [batch size, num cell, 1]
            object_relative_xy: [batch size, num cell, 1, 4]
            class_prob: [batch size, num cell, num classes]
            regression_coord_label: [batch size, num cell, 1, 4]
        '''
        with tf.name_scope('iou'):
            target_area = tf.reduce_prod(object_relative_xy[:,:,:,2:] - object_relative_xy[:,:,:,:2], axis=-1)
            # Element-wise maximum and minimum
            overlapped_xy_min = tf.maximum(self.area_xy_min, object_relative_xy[:,:,:,:2])
            overlapped_xy_max = tf.minimum(self.area_xy_max, object_relative_xy[:,:,:,2:])
            overlapped_area = tf.reduce_prod(overlapped_xy_max - overlapped_xy_min, axis=-1, name='overlapped_area')
    
            total_area = target_area + self.area - overlapped_area
            # [batch size, num cell, bounding boxes]
            self.iou = tf.truediv(overlapped_area, total_area, name='iou')

        # iou per bounding boxes (anchors), choose highest iou
        with tf.name_scope('mask'):
            # [batch size, num cells, 1]
            highest_iou = tf.reduce_max(self.iou, axis=2, keep_dims=True, name='highest_iou')
            # To get highest_iou index, use tf.equal(returns True/False for corresponding index)
            highest_iou_index = tf.to_float(tf.equal(self.iou, highest_iou), name='highest_iou_index')
            # Define confidnece as pr(object) * iou 
            # If no object exists in cell, the confidence scord would be zero otherw
            # Make one bounding box predictor which has highest iou to be responsible for each object
                # [batch size, num cells, bounding boxes]
            self.object_mask = highest_iou_index * object_appear
            self.no_object_mask = 1 - self.object_mask

        with tf.name_scope('regression_loss'):
            coordinate_obj = tf.reduce_sum(tf.expand_dims(self.object_mask, -1) * tf.square(self.coord - regression_coord_label))
            coordinate_objective = self.coord_hparam * coordinate_obj
            tf.summary.scalar('coordinate_objective', coordinate_objective)
            # confidence object's label should be 1 for reponsible bounding box
            confidence_obj = tf.reduce_sum(self.object_mask* tf.square(self.t_o - self.object_mask))
            confidence_noobj = tf.reduce_sum(self.no_object_mask * tf.square(self.t_o - self.object_mask))
            confidence_objective = confidence_obj + self.noobj_hparam * confidence_noobj
            tf.summary.scalar('confidence_objective', confidence_objective)
            probability_objective = tf.reduce_sum(tf.expand_dims(object_appear, -1) * tf.square(tf.expand_dims(class_prob, 2) - self.class_prob))
            tf.summary.scalar('probability_objective', probability_objective)

            total_objective = coordinate_objective + confidence_objective + probability_objective
            tf.summary.scalar('total_objective', total_objective)
    
            tf.add_to_collection(tf.GraphKeys.LOSSES, total_objective)
        

            
