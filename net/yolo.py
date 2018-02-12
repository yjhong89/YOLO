import tensorflow as tf
import tensorflow.contrib.slim as slim

def leaky_relu(x, alpha=0.1):
    with tf.name_scope('leaky_relu') as scope:
        return tf.maximum(x, x*alpha, name=scope)


# Fast YOLO, 9 layers
# Net must be 4 dimension
def yolo(net, is_training, classes=20, cell_width=7, cell_height=7, boxes_per_cell=2, name='yolo', channel=16, output_dim=4096):
    def batch_norm(net):
        net = slim.batch_norm(net, center=False, scale=True, epsilon=1e-5, is_training=is_training)
        # center is False
        net = tf.nn.bias_add(net, slim.variable('biases', shape=[net.get_shape().as_list()[-1]], initializer=tf.zeros_initializer()))
        return net
    with tf.variable_scope(name):
        # max_pool2d: stride=2(default), padding=SAME: output size is the same when when stride=1
        with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.01), normalizer_fn=batch_norm, activation_fn=leaky_relu), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2,2], padding='SAME'):
            layer_index = 0
            net = slim.layers.conv2d(net, channel, scope='conv%d' % (layer_index))
            net = slim.layers.max_pool2d(net, scope='max_pool%d' % (layer_index)) 
            layer_index += 1
            net = slim.layers.conv2d(net, channel*2, scope='conv%d' % (layer_index))
            net = slim.layers.max_pool2d(net, scope='max_pool%d' % (layer_index)) 
            layer_index += 1
            net = slim.layers.conv2d(net, channel*4, scope='conv%d' % (layer_index))
            net = slim.layers.max_pool2d(net, scope='max_pool%d' % (layer_index)) 
            layer_index += 1
            net = slim.layers.conv2d(net, channel*8, scope='conv%d' % (layer_index))
            net = slim.layers.max_pool2d(net, scope='max_pool%d' % (layer_index)) 
            layer_index += 1
            net = slim.layers.conv2d(net, channel*16, scope='conv%d' % (layer_index))
            net = slim.layers.max_pool2d(net, scope='max_pool%d' % (layer_index)) 
            layer_index += 1
            net = slim.layers.conv2d(net, channel*32, scope='conv%d' % (layer_index))
            net = slim.layers.max_pool2d(net, scope='max_pool%d' % (layer_index)) 
            layer_index += 1
            
            net = slim.layers.conv2d(net, channel*32, scope='conv%d' % ( layer_index))
            layer_index += 1
            net = slim.layers.conv2d(net, channel*64, scope='conv%d' % (layer_index))
            layer_index += 1
            net = slim.layers.conv2d(net, channel*16, scope='conv%d' % ( layer_index))
            # [batch size, 7, 7, 256]
            print(net.get_shape().as_list())
        # Flatten and Fully connected layer
        #_, grid_width, grid_height, _ = net.get_shape().as_list()
        net = slim.layers.flatten(net, scope='flatten')
        with slim.arg_scope([slim.layers.fully_connected], activation_fn=leaky_relu, normalizer_fn=batch_norm):
            layer_index = 0
            net = slim.layers.fully_connected(net, channel*16, scope='fc%d' %(layer_index))
            layer_index += 1
            net = slim.layers.fully_connected(net, output_dim, scope='fc%d' %(layer_index))
        # Final output: [batch size, 7,7, cell_width*cell_height*(boxes_per_cell*5+classes]]
        final_outputdim = cell_width * cell_height * (boxes_per_cell * 5 + classes)
        net = slim.layers.fully_connected(net, final_outputdim, scope='output')      

        return net


if __name__ == '__main__':
    net = tf.get_variable('test', [10, 448,448,3])
    a = yolo(net, False)
    print(a.get_shape())
