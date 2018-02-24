import tensorflow as tf
import tensorflow.contrib.slim as slim


def leaky_relu(x, alpha=0.1):
    with tf.name_scope('leaky_relu') as scope:
        return tf.maximum(x, x*alpha, name=scope)


def yolo2(net, is_training, num_anchors, classes, channel=32, name='yolo2'):
    def batch_norm(net):
        net = slim.batch_norm(net, center=True, scale=True, epsilon=1e-5, is_training=is_training)
        return net

    # Use 1*1 filters to compress the feature representation between 3*3 convolutions
    # Use batch normalization to stabilize training, speed up convergence, regularize the model
    with tf.variable_scope(name):
        with slim.arg_scope([slim.layers.conv2d], kernel_size=[3,3], stride=1, padding='SAME', normalizer_fn=batch_norm, activation_fn=leaky_relu), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2,2], stride=2, padding='SAME'):
            layer_index = 0
            for _ in range(2):
                net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
                print(net.get_shape().as_list())
                net = slim.layers.max_pool2d(net, scope='max_pool2d_%d' % layer_index)
                print(net.get_shape().as_list())
                channel *= 2
                layer_index += 1
            # channel=128, layer_index=2
            for _ in range(2):
                net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
                print(net.get_shape().as_list())
                layer_index += 1
                net = slim.layers.conv2d(net, channel/2, kernel_size=[1,1], scope='conv2d_%d' % layer_index)
                print(net.get_shape().as_list())
                layer_index += 1
                net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
                print(net.get_shape().as_list())
                net = slim.layers.max_pool2d(net, scope='max_pool2d_%d' % layer_index)
                print(net.get_shape().as_list())
                layer_index += 1
                channel *= 2
            # channel=512
            net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())
            net = slim.layers.conv2d(net, channel/2, kernel_size=[1,1], scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())
            net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())
            net = slim.layers.conv2d(net, channel/2, kernel_size=[1,1], scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())
            net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
            print(net.get_shape().as_list())
            '''
                For passthrough, we copy 26*26 resolution, (26,26,512)
                For localizing smaller objects, simply adding a passthrough layer that brings features from an earlier layer 
            '''
            pt = tf.identity(net, name='passthrough')
            net = slim.layers.max_pool2d(net, scope='max_pool2d_%d' % layer_index) 
            layer_index += 1
            channel *= 2

            # channel=1024
            net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())
            net = slim.layers.conv2d(net, channel/2, kernel_size=[1,1], scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())
            net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())
            net = slim.layers.conv2d(net, channel/2, kernel_size=[1,1], scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())
            net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())

            # Add three 3*3 convoultional layers with 1024 filters 
            net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())
            net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
            layer_index += 1
            print(net.get_shape().as_list())

            # passthrough layer concatenates (26,26,512) with (13,13,1024) by stacking adjacent features into different channels instead of spatial location
            ''' (6,6) -> (3,2,3,2) -> (3,3,4)
                [[1,2,3,4,5,6],      [[[1,2,5,6],
                 [6,5,4,3,2,1],        [3,4,4,3],
                 [1,2,3,4,5,6],        [5,6,2,1]]], 
                 [6,5,4,3,2,1], ->      ...
                 [1,2,3,4,5,6],       [[3,4,4,3],
                 [6,5,4,3,2,1]]        [5,6,2,1]]
            '''
            pt_shape = pt.get_shape().as_list()
            print('passthrough', pt_shape)
            with tf.name_scope('pass_through'):
                pt_net = tf.reshape(pt, [pt_shape[0], int(pt_shape[1]/2), 2, int(pt_shape[2]/2), 2, pt_shape[3]])
                pt_net = tf.transpose(pt_net, [0,1,3,2,4,5])
                pt_net = tf.reshape(pt_net, [pt_shape[0], int(pt_shape[1]/2), int(pt_shape[2]/2), pt_shape[3]*2*2])
                print(pt_net.get_shape().as_list())
            # pt_net: (13,13,2048)
            net = tf.concat([net, pt_net], axis=3, name='concat_pt')
            # Add a passthrough layer to the second to last convolutional layer
            net = slim.layers.conv2d(net, channel, scope='conv2d_%d' % layer_index)
            print(net.get_shape().as_list())
        # Remove fully connected layers, instead add final 1*1 convolustional layers with the number of outputs we need for detection
            # Predict boxes with 5 coordinates each and 20 classes per box -> 125 filters
        net = slim.layers.conv2d(net, num_anchors*(5+classes), kernel_size=[1,1], activation_fn=None, scope='final')
        print(net.get_shape().as_list())

        return net


if __name__ == '__main__':
    net = tf.get_variable('test', [10, 416,416,3])
    a = yolo2(net, False, 5, 20)
    for a in tf.trainable_variables():
        print(a.op.name)        

