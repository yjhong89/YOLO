import tensorflow as tf
import os

# In tfrecord, 
# Image name(bytes), Image size(int), Object info(Object class, Object coord)
 
def read_tfrecord(tfrecord_path, config):
    # tf.name_scope is for operators
    with tf.name_scope(tfrecord_path.split('.')[0]):
        # Read tfrecord file
        # Create a queue to hold filenames using tf.train.string_input_produce,
            # hold filenames in a FIFO queue(list)
        file_queue = tf.train.string_input_producer([tfrecord_path], shuffle=True)
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
        with tf.name_scope('decoded'):
            object_class = tf.decode_raw(decoded_example['object_info'][0], tf.int64, name='object_class')
            object_coord = tf.decode_raw(decoded_example['object_info'][1], tf.float32, name='object_coord')
            # Need to reshape-> shape is [None]
            object_coord = tf.reshape(object_coord, [-1,4])
            image_shape = tf.cast(decoded_example['image_size'], tf.int64, name='image_shape')
            # tf.read_file(filename): filename is a tensor of type string and outputs the contents of filename
            image_file = tf.read_file(decoded_example['image_path'])
            image = tf.image.decode_jpeg(image_file, channels=3)
            image = tf.cast(image, tf.float32)

    model_name = config.get('config', 'model')

    # Image_shape: height, width, depth (375, 500, 3)
        # We neet to resize image to (448,448,3)
    resized_image, resized_object_coord = resize_image(image, image_shape, object_coord, config.getint(model_name, 'width'), config.getint(model_name, 'height'))
    image = tf.clip_by_value(resized_image, 0, 255)

    down_ratio = int(config.getint(model_name, 'width') / config.getint(model_name, 'cell_width'))
    processed_label = label_processing(object_class, num_class, resized_object_coord, config.getint(model_name, 'cell_width'), config.getint(model_name, 'cell_height'), down_ratio)

    return image, processed_label

def resize_image(image, image_shape, object_coord, config_width, config_height):
    # To do division
    raw_image_height = tf.cast(image_shape[0], tf.float32)
    raw_image_width = tf.cast(image_shape[1], tf.float32)
    with tf.name_scope('resize'):
        resized_image = tf.image.resize_images(image, [config_width, config_height])
        # shape of [2,], multiply resize_factor as width to 'x', height to 'y'
        resize_factor = [config_width/raw_image_width, config_height/raw_image_height]
        # tf.tile([a,b,c],[2]): [a,b,c,a,b,c]
        resized_object_coord = object_coord * tf.tile(resize_factor, [2])

    return resized_image, resized_object_coord
    
# Process object class and object coordination information
# Normalize the bounding box width and height by the image width and height so that they fall between 0 and 1.
# (x,y) coordinates represent the center of the box relative to the bounds of the grid cell
    # (x,y) becomes the offset of a particular grid cell
def label_processing(object_class, num_class, object_coord, cell_width, cell_height, ratio):
    if len(object_class) != len(object_coord):
        raise ValueError('Number of object is not same')

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
    object_cell_index = (np.floor(object_cell_y) * cell_width + np.floor(object_cell_x)).astype(np.int)
    # offset between cell boundary and obejct center
    offset_x = object_cell_x - np.floor(object_cell_x)
    offset_y = object_cell_y - np.floor(object_cell_y)
    # width and height are predicted relative to the whole image
    object_width = ((x_max - x_min) / ratio) / cell_width
    object_height = ((y_max - y_min) /ratio) / cell_height

    # [num_cell, 1, *]: middle '1' is for 'boxes_per_cell'

    # To calculate regression problem, pass coordinate(x,y,w,h)
    regression_coord_label = np.zeros([num_cell,1, 4], dtype=np.float32)
    regression_coord_label[object_cell_index,0, 0] = offset_x
    regression_coord_label[object_cell_index,0, 1] = offset_y
    # width and height regression is square root
    regression_coord_label[object_cell_index,0, 2] = np.sqrt(object_width)
    regression_coord_label[object_cell_index,0, 3] = np.sqrt(object_height)

    # Object appear 
    object_appear = np.zeros([num_cell, 1], dtype=np.int)
    object_appear[object_cell_index] = 1

    # Each cell prdicts conditional class probabilities pr(class|object), conditioned on the cell containing an object
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
