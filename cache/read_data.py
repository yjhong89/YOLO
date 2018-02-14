import os
import tensorflow as tf
import numpy as np
# Library parsing html file
from bs4 import BeautifulSoup

def check_file_list(file_list):
    checklist = list(filter(os.path.exists, file_list))
    if len(checklist) != len(file_list):
        tf.logging.warn('Some images donot exists')
        return False
    else:
        return True

def xml_parsing(xml_path, class_index):
    with open(xml_path, 'r') as f:
        # f.read() returns all file contents
        parsed = BeautifulSoup(f.read(), 'xml').find('annotation')

    object_index = list()
    object_coord = list()
       
    # recursive=False: search only direct children
    for obj in parsed.find_all('object', recursive=False):
    # In obj: name, pose, truncated, difficult, bndbox
        for name, bndbox, in zip(obj.find_all('name', recursive=False), obj.find_all('bndbox', recursive=False)):
            if name.string in class_index:
                object_index.append(class_index[name.string])
                # image index starts with 0
                xmin = float(bndbox.xmin.string) - 1
                xmax = float(bndbox.xmax.string) - 1
                ymin = float(bndbox.ymin.string) - 1
                ymax = float(bndbox.ymax.string) - 1
                object_coord.append((xmin, ymin, xmax, ymax))
            else:
                raise IndexError('Name does not exist')
    size_info = parsed.find('size')
    size = (int(size_info.height.string), int(size_info.width.string), int(size_info.depth.string))
    image_name = parsed.find('filename').text
    #print(image_name, size, object_index, object_coord)
    return image_name, size, object_index, object_coord
    
# From pandas, row['key'] returns corresponding value
def voc(writer, class_index, data_type, row, basedir, verify=False):
    # row['root']: indexing with header 'root'
    voc_path = os.path.join(basedir, row['root'])
    data_type_path = os.path.join(voc_path, 'ImageSets', 'Main', data_type + '.txt')
    # 'test' does not exist
    if not os.path.exists(data_type_path):
        tf.logging.warn('No ' + data_type_path)
        return False

    with open(data_type_path, 'r') as f:
        filename = [line.strip() for line in f]
    # Access to xml files (Annotation)
    annotations = [os.path.join(voc_path, 'Annotations', xml_file + '.xml') for xml_file in filename]
    if check_file_list(annotations) is not True:
        raise Exception('File not match')
    
    # Number of images, which do not contain objects
    count_noobj = 0
    for path in annotations:
        image_name, image_size, object_index, object_coord = xml_parsing(path, class_index)
        # If noobj, ther is no 'name' for 'object' in xml file
        if len(object_index) == 0:
            count_noobj += 1
        object_index = np.asarray(object_index, dtype=np.int64)
        object_coord = np.asarray(object_coord, dtype=np.float32)
        image_path = os.path.join(voc_path, 'JPEGImages', image_name)
        print(image_path)

        # Here, we create a tfrecord file
        
        # Convert data into proper data type of the feature using 'tf.train.Int64List', 'tf.train.BytesList', 'tf.train.FloatList': these function expects 'value' to be either a list or numpy array    
        # Create a feature using 'tf.train.Feature'
        features = {'image_path' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(image_path)]))}
            # tf.compat.as_bytes(bytes or text): Convert input string to bytes
            # value=: constructino of array is necessary-> value=[...] to make array
        features['image_size'] = tf.train.Feature(int64_list=tf.train.Int64List(value=image_size))
            # value=image_size: list ('[image_size]') is not allowed since image_size is already tuple, so remove the constriction of the array
        features['object_info'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[object_index.tostring(), object_coord.tostring()]))
            # numpy array.tostring(): Convert to bytes
        # Create an Example protocol buffer using 'tf.train.Example'
        example = tf.train.Example(features=tf.train.Features(feature=features))
        # Serialize the Example to string using example.SerializToString() and write the serialized example to tfrecord file
        writer.write(example.SerializeToString())
    if count_noobj > 0:
        tf.logging.warn('%d images do not contain objects' % (count_noobj))

    return True
        
