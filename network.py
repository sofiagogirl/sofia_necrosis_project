import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import ops, sys

def conv2d(inp, shp, name, strides=(1,1,1,1), padding='SAME', trainable=True):
    with tf.device('/cpu:0'):
        filters = tf.get_variable(name + '/filters', shp, initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/(shp[0]*shp[1]*shp[3]))), trainable=trainable)
        biases = tf.get_variable(name + '/biases', [shp[-1]], initializer=tf.constant_initializer(0), trainable=trainable)
    return tf.nn.bias_add(tf.nn.conv2d(inp, filters, strides=strides, padding=padding), biases)

def conv2d_d(inp, shp, name, strides=(1,1,1,1), padding='SAME', trainable=True):
    with tf.device('/cpu:0'):
        w = tf.get_variable(name + '/filters', shp, trainable=trainable)
       # filters = tf.get_variable(name + '/filters', shp, initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/(shp[0]*shp[1]*shp[3]))), trainable=trainable)
        biases = tf.get_variable(name + '/biases', [shp[-1]], initializer=tf.constant_initializer(0), trainable=trainable)
    return tf.nn.bias_add(tf.nn.conv2d(inp, filter=spectral_norm(w,name), strides=strides, padding=padding), biases)

def spectral_norm(w,name, iteration=1):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable(name+'u', [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)

   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)

   return w_norm

def leakyRelu(x, alpha=0.1):
    xx = tf.layers.batch_normalization(x)
    return tf.nn.relu(xx) - alpha * tf.nn.relu(-xx)

def leakyRelu_d(x, alpha=0.1):
   # xx = tf.layers.batch_normalization(x)
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def fc_layer(inp, shp, name):
    with tf.device('/cpu:0'):
        #weights = tf.get_variable(name + '/weights', shp, initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.get_variable(name + '/weights', shp, initializer=tf.initializers.glorot_uniform())
        biases = tf.get_variable(name + '/biases', [shp[-1]], initializer=tf.constant_initializer(0))
    return tf.nn.bias_add(tf.matmul(inp, weights), biases)

def normal_block(inp, name, is_training):
    ch = inp.get_shape().as_list()[-1]
    conv1 = leakyRelu_d(conv2d_d(inp, [3,3,ch,ch], name + '/conv1'))
    conv2 = leakyRelu_d(conv2d_d(conv1, [3,3,ch,ch*2], name + '/conv2', strides=(1,2,2,1)))
    return conv2

def _YCbCr2RGB(image):
    X = image[...,0] + 1.403* (image[...,1] - 128)
    Y = image[...,0] - 0.714* (image[...,1] - 128) - 0.344*(image[...,2] - 128)
    Z = image[...,0] + 1.773* (image[...,2] - 128)
    X = tf.expand_dims(X, axis=3)
    Y = tf.expand_dims(Y, axis=3)
    Z = tf.expand_dims(Z, axis=3)
    return tf.concat([X, Y, Z], axis=3)

def _normalize(image):
    X = (image[...,0] - tf.reduce_min(image[...,0]))/( tf.reduce_max(image[...,0])- tf.reduce_min(image[...,0]))*255.0
    Y = (image[...,1] - tf.reduce_min(image[...,1]))/( tf.reduce_max(image[...,1])- tf.reduce_min(image[...,1]))*255.0
    Z = (image[...,2] - tf.reduce_min(image[...,2]))/( tf.reduce_max(image[...,2])- tf.reduce_min(image[...,2]))*255.0
    X = tf.expand_dims(X, axis=3)
    Y = tf.expand_dims(Y, axis=3)
    Z = tf.expand_dims(Z, axis=3)
    return tf.concat([X, Y, Z], axis=3)

def sobelFilter(inp):
    inp = tf.greater(inp, 0.05)
    inp = tf.cast(inp, tf.float32)
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    filtered_x = tf.nn.conv2d(inp, sobel_x_filter,
                              strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(inp, sobel_y_filter,
                              strides=[1, 1, 1, 1], padding='SAME')
    return tf.cast(tf.cast(tf.abs(filtered_x) + tf.abs(filtered_y), tf.bool), tf.float32)

class Generator(object):

    def __init__(self, inp, config):
        self.dic = {}
        self.config = config
        cur = inp
        print(cur.get_shape())
        for i in range(self.config.n_levels):
            cur = self.down(cur, i)
        ch = cur.get_shape().as_list()[-1]
        cur = leakyRelu(conv2d(cur, [3,3,ch,ch], 'Gen_center'))
        for i in range(self.config.n_levels):
            cur = self.up(cur, self.config.n_levels - i - 1)

        self.output = conv2d(cur, [3,3,self.config.n_channels//2,3], 'Gen_last_layer')

    def down(self, inp, lvl):
        name = 'Gen_down{}'.format(lvl)
        in_ch = inp.get_shape().as_list()[-1]
        out_ch = self.config.n_channels if lvl == 0 else in_ch * 2
        mid_ch = (in_ch + out_ch) // 2
        conv1 = leakyRelu(conv2d(inp, [3,3,in_ch,mid_ch], name + '/conv1'))
        conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
        conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
        tmp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,out_ch-in_ch]], 'CONSTANT')
        self.dic[name] = conv3 + tmp
        # return conv2d(self.dic[name], [3,3,out_ch,out_ch], name + '/downsampling_conv', strides=(1,2,2,1))
        return tf.nn.avg_pool(self.dic[name], ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

    def up(self, inp, lvl):
        name = 'Gen_up{}'.format(lvl)
        size = self.config.image_size >> lvl
        image = tf.image.resize_bilinear(inp, [size, size])
        image = tf.concat([image, self.dic[name.replace('up', 'down')]], axis=3)
        in_ch = image.get_shape().as_list()[-1]
        out_ch = in_ch // 4
        mid_ch = (in_ch + out_ch) // 2
        conv1 = leakyRelu(conv2d(image, [3,3,in_ch,mid_ch], name + '/conv1'))
        conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
        conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
        return conv3

class Discriminator(object):
    def __init__(self, inp, config):
        cur = leakyRelu_d(conv2d_d(inp, [3,3,3,config.n_channels], 'conv1'))
        for i in range(config.n_blocks):
            cur = normal_block(cur, 'n_block{}'.format(i), config.is_training)
        cur = tf.reduce_mean(cur, axis=(1,2))
        ch = cur.get_shape().as_list()[-1]
        cur = leakyRelu_d(fc_layer(cur, [ch, ch], 'fcl1'))
        self.output = tf.nn.sigmoid(fc_layer(cur, [ch, 1], 'fcl2'))

class UNet(object):
    def __init__(self, inp, config, trainable=False):
        print(inp.shape)
        self.dic = {}
        self.config = config
        down1 = self.down(inp,   3,   32, 'down1', trainable)
        down2 = self.down(down1, 32,  64, 'down2', trainable)
        down3 = self.down(down2, 64,  128, 'down3', trainable)

        ctr = tf.nn.relu(conv2d(down3, [3,3,128,128], 'center', trainable= trainable))

        size = self.config.image_size
        up3 = self.up(ctr, 128*2, 64, size//4, 'up3', trainable)
        up2 = self.up(up3, 64*2, 32,  size//2, 'up2', trainable)

        up1 = self.up(up2, 32*2, 32,  size, 'up1', trainable)
        self.output = tf.nn.sigmoid(conv2d(up1, [3,3,32,1], 'last_layer', trainable=trainable))

    def down(self, inp, in_ch, out_ch, name, trainable):
        mid_ch = (in_ch + out_ch) // 2
        conv1 = tf.nn.relu(conv2d(inp, [3,3,in_ch,mid_ch], name + '/conv1', trainable=trainable))
        conv2 = tf.nn.relu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2', trainable=trainable))
        conv3 = tf.nn.relu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3', trainable=trainable))
        # if in_ch == 3:
        #     tf.add_to_collection('hidden_loss', conv3[..., 26])
        tmp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,out_ch-in_ch]], 'CONSTANT')
        self.dic[name] = conv3 + tmp
        return tf.nn.max_pool(self.dic[name], ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

    def up(self, inp, in_ch, out_ch, size, name, trainable):
        image = tf.image.resize_bilinear(inp, [size, size])
        image = tf.concat([image, self.dic[name.replace('up', 'down')]], axis=3)
        mid_ch = (in_ch + out_ch) // 2
        conv1 = tf.nn.relu(conv2d(image, [3,3,in_ch,mid_ch], name + '/conv1', trainable=trainable))
        conv2 = tf.nn.relu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2', trainable=trainable))
        conv3 = tf.nn.relu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3', trainable=trainable))
        return conv3
