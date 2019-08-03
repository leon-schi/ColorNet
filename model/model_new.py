import tensorflow as tf
tf.keras.backend.clear_session()

from tensorflow import keras
from.color_mapping import ColorEncoder

class ColorNetBuilder:
    def __init__(self, input_shape, 
                    output_dimensions=ColorEncoder.vec_size,
                    initial_num_filters=64,
                    num_poolings=0,
                    kernel_size=3,
                    dropout_rate=0.2,
                    batch_normalization=True):
        self.output_dimensions = output_dimensions
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization

        self.activation = 'relu'
        self.padding = 'same'
        self.kernel_initializer = 'glorot_uniform'

        self.model_name = 'u-net'

    def conv2d_layer(self, inputs, num_filters=None, kernel_size=None, activation=None, strides=None):
        """
        builds a custom 2d convolutional layer based on the current configuration.
        """
        kernel_size = self.kernel_size if (kernel_size == None) else kernel_size
        num_filters = self.initial_num_filters if (num_filters == None) else num_filters
        activation = self.activation if (activation == None) else activation
        strides = (1, 1) if (strides == None) else strides

        x = keras.layers.Conv2D(num_filters, kernel_size, 
                                    padding=self.padding, 
                                    kernel_initializer=self.kernel_initializer, 
                                    strides=strides)(inputs)
        return keras.layers.LeakyReLU(alpha=0.02)(x)

    def build_downsampling_block(self, inputs, num_filters, stride=1):
        """
        builds a block in the downsampling phase of the net.
        It consists of 2 Conv2D layers with batch normalization and max pooling in the end.
        The result before pooling is also returned, to pass it on to the upsampling phase
        """
        x = self.conv2d_layer(inputs, num_filters=num_filters)
        x = self.conv2d_layer(x, num_filters=num_filters, strides=(stride, stride))
        x = keras.layers.Dropout(rate=self.dropout_rate)(x)
        return keras.layers.BatchNormalization()(x) if (self.batch_normalization) else x

    def build_upsampling_block(self, inputs, num_filters):
        """
        builds a block in the upsampling phase of the net.
        It consists of one Deconvolution (upsampling) layer with
        concatenation of inputs from an earlier step and two Conv2D layers
        """
        x = keras.layers.UpSampling2D(size = (2,2))(inputs)
        x = self.conv2d_layer(x, num_filters=num_filters)
        x = self.conv2d_layer(x, num_filters=num_filters)
        x = keras.layers.Dropout(rate=self.dropout_rate)(x)
        return keras.layers.BatchNormalization()(x) if (self.batch_normalization) else x

    def build_output_block(self, inputs, num_filters):
        x = self.conv2d_layer(inputs, num_filters=num_filters)
        return keras.layers.Conv2D(self.output_dimensions, self.kernel_size,
                                    activation='softmax', 
                                    padding=self.padding, 
                                    kernel_initializer=self.kernel_initializer, 
                                    strides=(1,1))(x)

    def build_model_core(self):
        """
        builds the core of the u-net model with the downsampling and upsampling stages
        """
        self.inputs = keras.Input(shape=self.input_shape, name='bw-img')
        
        x = self.build_downsampling_block(self.inputs, 64, 2)  #128x128
        x = self.build_downsampling_block(x, 128, 2) #64x64
        x = self.build_downsampling_block(x, 256, 2) #32x32
        x = self.build_downsampling_block(x, 512, 1) #32x32
        x = self.build_downsampling_block(x, 512, 1) #32x32

        x = self.build_upsampling_block(x, 256)      #64x64
        x = self.build_upsampling_block(x, 128)      #128x128
        self.outputs = self.build_output_block(x, 256)          #128x128

    def instanciate_model(self):
        return keras.Model(inputs=self.inputs, outputs=self.outputs, name=self.model_name)

    def calculate_output_shape(self, input_shape):
        assert input_shape[0] % 2 == 0
        assert input_shape[1] % 2 == 0
        return (int(input_shape[0]/2), int(input_shape[1]/2), self.outputs.shape[-1].value)

    def build_model(self):
        self.build_model_core()
        return self.instanciate_model()