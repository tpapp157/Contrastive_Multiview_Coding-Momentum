from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf



class ConvNorm(Layer):

    def __init__(self, filters, kernel_size=3, strides=1, kernel_initializer='glorot_uniform', **kwargs):
        super(ConvNorm, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = strides
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape, initializer=self.kernel_initializer, name='kernel')
        self.built = True

    def call(self, inputs):
        weights = self.kernel
        
        x = inputs

        #Normalize
        d = K.sqrt(K.sum(K.square(weights), axis=[0,1,2], keepdims=True) + 1e-8)
        weights = weights / d
        
        if self.kernel_size[0]>1:
            p = (self.kernel_size[0]-1)//2
            x = tf.pad(x, [[0,0], [p,p], [p,p], [0,0]], mode='SYMMETRIC')
        
        x = tf.nn.conv2d(x, weights, strides=self.strides, padding="VALID")
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape[:3] + self.out_channels
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))