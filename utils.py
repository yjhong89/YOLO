import tensorflow as tf
import os
import numpy as np


def image_per_standardization(image):
    # tf.image.per_image_standardization
    normalized_image = (image - np.mean(image)) / max(np.std(image), 1/np.sqrt(np.prod(image.shape[1:])))
    return normalized_image

# In tfrecord, 
# Image name(bytes), Image size(int), Object info(Object class, Object coord)
def read_tfrecord(tfrecord_path, config, num_class):
    # tf.name_scope is for operators
    with tf.name_scope('read_tfrecord'):
        # Read tfrecord file
        # Create a queue to hold filenames using tf.train.string_input_produce,
            # hold filenames in a FIFO queue(list)
        file_queue = tf.train.string_input_producer(tfrecord_path, shuffle=True)
        # Define a reader, reader returns the next record using reader.read(filename_queue)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)
        # Define a decoder: tf.parse_single_example, it takes a serialized example and a dictionary which maps feature keys to FixedLenFeature and returns a dictionary which maps feature keys to tensor
            # []: the number of feature
        features = {'image_path' : tf.FixedLenFeature([], tf.string)}
        features['image_size'] = tf.FixedLenFeature([3], tf.int64)
        features['object_info'] = tf.FixedLenFeature([2], tf.string)
        decoded_example = tf.parse_single_example(serialized_example, features=features) 
        # Convert the data from string back to the number
            # tf.decode_raw(bytes, out_type) takes a tensor of type string and convert it to 'out_type'
            # For labels-> tf.cast
        with tf.name_scope('decode'):
            object_class = tf.decode_raw(decoded_example['object_info'][0], tf.int64, name='object_class')
            object_coord = tf.decode_raw(decoded_example['object_info'][1], tf.float32, name='object_coord')
            # Need to reshape-> shape is [None], [xmin, ymin, xmax, ymax]
            object_coord = tf.reshape(object_coord, [-1,4])
            # height, width. depth
            image_shape = tf.cast(decoded_example['image_size'], tf.int64, name='image_shape')
            # tf.read_file(filename): filename is a tensor of type string and outputs the contents of filename
            image_file = tf.read_file(decoded_example['image_path'])
            # uint8, 
            image = tf.image.decode_jpeg(image_file, channels=3)
            image = tf.cast(image, tf.float32)

    model_name = config.get('config', 'model')

    if config.getboolean('augmentation', 'crop'):
        image, object_coord, image_shape = tf.cond(tf.random_uniform([], maxval=1.0) < config.getfloat('augmentation', 'probability'),
            lambda: random_crop(image, image_shape, object_coord),
            lambda: (image, object_coord, image_shape))
        
    # Image_shape: height, width, depth (375, 500, 3)
        # We neet to resize image to (448,448,3)
    resized_image, resized_object_coord = resize_image(image, image_shape, object_coord, config.getint(model_name, 'width'), config.getint(model_name, 'height'))
    
    # Adjust the saturation of an RGB image by a random factor
    if config.getboolean('augmentation', 'saturation'):
        resized_image = tf.cond(tf.random_uniform([], maxval=1.0) < config.getfloat('augmentation', 'probability'),
            lambda: tf.image.random_saturation(resized_image, lower=0.5, upper=1.5),
            lambda: resized_image)
    
    image = tf.clip_by_value(resized_image, 0, 255)

    cell_width = config.getint(model_name, 'cell_width')
    cell_height = config.getint(model_name, 'cell_height')
    num_cells = cell_width * cell_height

    down_ratio = int(config.getint(model_name, 'width') / config.getint(model_name, 'cell_width'))
    tf.logging.info('Down sampling ratio %d' % down_ratio)
    ''' 
        labels from tfrecord file are Tensor object.
        But we want to process them as numpy arrays (Tensor object is not iterable) 
        -> use tf.py_func
    '''    
    '''
        tf.py_func(func, input, Tout): takes numpy array and returns numpy array as its output, wrap this function as anoperation in a tensorflow graph.
            func : A python function, which takes numpy array havin gelement types that match Tensor object in 'inp'
            inp : A list of Tensor object for 'func''s arguments
            Tout : A list of tensorflow data type which indicates what 'func' return
            return : A list of Tensor which func computes            
    '''
    object_appear, object_relative_xy, class_prob, regression_coord_label = tf.py_func(label_processing, [object_class, num_class, resized_object_coord, cell_width, cell_height, down_ratio], [tf.float32] * 4) 
    # tf.py_func returns unknown shape-> need to reshape return values
    with tf.name_scope('reshaping_label'):
        object_appear = tf.reshape(object_appear, [num_cells, 1])
        object_relative_xy = tf.reshape(object_relative_xy, [num_cells, 1, 4])
        class_prob = tf.reshape(class_prob, [num_cells, num_class])
        regression_coord_label = tf.reshape(regression_coord_label, [num_cells, 1, 4])
    #processed_label = label_processing(object_class, num_class, resized_object_coord, config.getint(model_name, 'cell_width'), config.getint(model_name, 'cell_height'), down_ratio)
    
    labels = (object_appear, object_relative_xy, class_prob, regression_coord_label)

    return image, labels

def resize_image(image, image_shape, object_coord, config_width, config_height):
    # To do division
    raw_image_height = tf.cast(image_shape[0], tf.float32)
    raw_image_width = tf.cast(image_shape[1], tf.float32)
    with tf.name_scope('resize'):
        resized_image = tf.image.resize_images(image, [config_height, config_width])
        # shape of [2,], multiply resize_factor as width to 'x', height to 'y'
        resize_factor = [config_width/raw_image_width, config_height/raw_image_height]
        # tf.tile([a,b,c],[2]): [a,b,c,a,b,c]
        resized_object_coord = object_coord * tf.tile(resize_factor, [2])

    return resized_image, resized_object_coord

# For data augmentation
    # object_coord: (xmin, ymin, xmax, ymax)
    # image_shape: {height, width, depth)
def random_crop(image, image_shape, object_coord):
    with tf.name_scope('random_crop'):
        # Get xymin, xymax
        xy_min = tf.reduce_min(object_coord[:,:2], axis=0)
        xy_max = tf.reduce_max(object_coord[:,2:], axis=0)
        # image_shape[1::-1]: (width, height)
        max_margin = image_shape[1::-1] - xy_max
        shrink_ratio = tf.random_uniform([4], minval=0, maxval=1.0) * tf.concat([xy_min, max_margin], axis=0)

        cropped_object_coord = object_coord - tf.tile(shrink_ratio[:2], [2])
        cropped_width_height = image_shape[1::-1] - shrink_ratio[:2] - shrink_ratio[2:]

        cropped_image = tf.image.crop_to_bounding_box(image, tf.cast(shrink_ratio[0], tf.int32), tf.cast(shrink_ratio[1], tf.int32), tf.cast(cropped_width_height[0], tf.int32), tf.cast(cropped_width_height[1], tf.int32))

        return cropped_image, cropped_object_coord, cropped_width_height

def rotate(image, object_coord, degree, image_center_x, image_center_y):
    with tf.name_scope('rotate'):
        rotate_image = tf.contrib.image.rotate(image, degree, interpolation='NEAREST')
        # Make rotate matrix
            # Use tf.dynamic_stitch(indices, value) to make matrix
        rotate_mtx = tf.dynamic_stitch([[0],[1],[2],[3]], [tf.cos(degree), tf.sin(degree), -tf.sin(degree), tf.cos(degree)])
        rotate_mtx = tf.reshape(rotate_mtx, [2,2])
        image_center = np.reshape(np.array([image_center_x, image_center_y]), (2,1))
        rotated_coord_xy_min = tf.matmul(rotate_mtx, tf.transpose(object_coord[:,:2], [1,0]) - image_center) + image_center
        rotated_coord_xy_max = tf.matmul(rotate_mtx, tf.transpose(object_coord[:,2:], [1,0]) - image_center) + image_center
        # Note that xymin and xymax has been changed
        # Back to original shape
        rotated_coord_xy_min = tf.transpose(rotated_coord_xy_min, [1,0])
        rotated_coord_xy_max = tf.transpose(rotated_coord_xy_max, [1,0])
        xy_min = tf.minimum(rotated_coord_xy_min, rotated_coord_xy_max)
        xy_max = tf.maximum(rotated_coord_xy_min, rotated_coord_xy_max)
        rotate_object_coord = tf.concat([xy_min, xy_max], axis=1)

        return rotate_iamge, rotate_object_coord
    
# Process object class and object coordination information
# Normalize the bounding box width and height by the image width and height so that they fall between 0 and 1.
# (x,y) coordinates represent the center of the box relative to the bounds of the grid cell
    # (x,y) becomes the offset of a particular grid cell
def label_processing(object_class, num_class, object_coord, cell_width, cell_height, ratio):

    assert len(object_class) == len(object_coord)

    # All arguments are Tensor type, need to calculate based on numpy array

    # Divde input image into s*s cell. If the center of an object fall into a cell, that cell is reponsible for detecting object
    num_cell = cell_width * cell_height
    # object_coord: [num_of_object, 4] -> [4, num_of_object] by transpose
        # Each has shape of [num_of_object,]
    x_min, y_min, x_max, y_max = object_coord.T 
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    # (n,m) cell, (448, 448) -> (7,7), ratio: 64, each cell [0,1]
    object_cell_x = x_center / ratio
    object_cell_y = y_center / ratio
    # Calculate cell index, shape of [num_of_object,]
        # Only size-1 arrays can be converted to python scalar, int()
        # -> .astype(np.int)
    object_cell_index = (np.floor(object_cell_y) * cell_width + np.floor(object_cell_x)).astype(np.int)
    # offset between cell boundary and obejct center
    offset_x = object_cell_x - np.floor(object_cell_x)
    offset_y = object_cell_y - np.floor(object_cell_y)
    # width and height are predicted relative to the whole image
    object_width = ((x_max - x_min) / ratio) / cell_width
    object_height = ((y_max - y_min) /ratio) / cell_height

    #print('Index', object_cell_index)
    #print('width', object_width)
    #print('height', object_height)
    #print('x', offset_x) 
    #print('y',offset_y)

    # [num_cell, 1, *]: middle '1' is for 'boxes_per_cell'

    # To calculate regression problem, pass coordinate(x,y,w,h)
    regression_coord_label = np.zeros([num_cell,1, 4], dtype=np.float32)
    regression_coord_label[object_cell_index,0, 0] = offset_x
    regression_coord_label[object_cell_index,0, 1] = offset_y
    # width and height regression is square root
    regression_coord_label[object_cell_index,0, 2] = np.sqrt(object_width)
    regression_coord_label[object_cell_index,0, 3] = np.sqrt(object_height)

    # Object appear 
    object_appear = np.zeros([num_cell, 1], dtype=np.float32)
    object_appear[object_cell_index] = 1

    # Each cell prdicts conditional class probabilities pr(class|object), conditioned on the cell containing an object
    # Two bounding box share conditional probability of object
    class_prob = np.zeros([num_cell, num_class], dtype=np.float32)
    # object_class.shape = [num_of_object,]
    class_prob[object_cell_index, object_class] = 1

    # To calculate Intersection Over Union(IOU), we need area information.
    # Note that area is normally larger than one cell and each cell are normalized to [0,1], area is over the one grid-> respect to (7,7)
    object_relative_xy = np.zeros([num_cell,1, 4], dtype=np.float32)
    object_relative_xy[object_cell_index,0, 0] = offset_x - (object_width / 2) * cell_width
    object_relative_xy[object_cell_index,0, 1] = offset_y - (object_height / 2) * cell_height
    object_relative_xy[object_cell_index,0, 2] = offset_x + (object_width / 2) * cell_width
    object_relative_xy[object_cell_index,0, 3] = offset_y + (object_height / 2) * cell_height

    return object_appear, object_relative_xy, class_prob, regression_coord_label
        

if __name__ == "__main__":
    a,b,c,d = read_tfrecord('./cache/tfrecord/train.tfrecord')        
