
import tensorflow as tf
tf.keras.backend.clear_session()

from tensorflow import keras
from.color_mapping import ColorEncoder

class UNetBuilder:
    def __init__(self, input_shape, output_dimensions,
                    initial_num_filters=64, 
                    num_poolings=4, 
                    kernel_size=3,
                    dropout_rate=0.2,
                    batch_normalization=True):
        self.output_dimensions = output_dimensions
        self.input_shape = input_shape
        self.initial_num_filters = initial_num_filters
        self.num_poolings = num_poolings
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization

        self.activation = 'relu'
        self.padding = 'same'
        self.kernel_initializer = 'he_normal'

        self.model_name = 'u-net'

    def conv2d_layer(self, inputs, num_filters=None, kernel_size=None, activation=None):
        """
        builds a custom 2d convolutional layer based on the current configuration.
        """
        kernel_size = self.kernel_size if (kernel_size == None) else kernel_size
        num_filters = self.initial_num_filters if (num_filters == None) else num_filters
        activation = self.activation if (activation == None) else activation
        
        return keras.layers.Conv2D(num_filters, kernel_size, activation=activation, padding=self.padding, kernel_initializer=self.kernel_initializer)(inputs)

    def build_downsampling_block(self, inputs, num_filters):
        """
        builds a block in the downsampling phase of the net.
        It consists of 2 Conv2D layers with batch normalization and max pooling in the end.
        The result before pooling is also returned, to pass it on to the upsampling phase
        """

        x = self.conv2d_layer(inputs, num_filters=num_filters)
        x = self.conv2d_layer(x, num_filters=num_filters)
        before_pooling = keras.layers.BatchNormalization()(x) if (self.batch_normalization) else x
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(before_pooling)
        
        return x, before_pooling

    def build_lowest_convolution_block(self, inputs, num_filters):
        """
        builds the convolutional block at the lowest level and applies Dropout
        """
        x = self.conv2d_layer(inputs, num_filters=num_filters)
        x = self.conv2d_layer(x, num_filters=num_filters)
        x = keras.layers.BatchNormalization()(x) if (self.batch_normalization) else x
        return keras.layers.Dropout(rate=self.dropout_rate)(x)

    def upsampling_and_concatenation_layer(self, inputs, earlier_inputs, num_filters):
        """
        builds a layer that performs the upsampling (deconvolution layer) and
        concatenates the outputs with the results from the associated step in the downsampling phase.
        """
        x = keras.layers.UpSampling2D(size = (2,2))(inputs)
        x = self.conv2d_layer(x, num_filters=num_filters, kernel_size=2)
        return keras.layers.concatenate([x, earlier_inputs], axis = 3)

    def build_upsampling_block(self, inputs, earlier_inputs, num_filters):
        """
        builds a block in the upsampling phase of the net.
        It consists of one Deconvolution (upsampling) layer with
        concatenation of inputs from an earlier step and two Conv2D layers
        """
        x = self.upsampling_and_concatenation_layer(inputs, earlier_inputs, num_filters)
        x = self.conv2d_layer(x, num_filters=num_filters)
        x = self.conv2d_layer(x, num_filters=num_filters)
        x = keras.layers.BatchNormalization()(x) if (self.batch_normalization) else x
        return keras.layers.Dropout(rate=self.dropout_rate)(x)

    def build_model_core(self):
        """
        builds the core of the u-net model with the downsampling and upsampling stages
        """
        self.inputs = keras.Input(shape=self.input_shape, name='bw-img')
        num_filters = self.initial_num_filters
        x = self.inputs
        
        # build the downsampling stage
        self.down = dict()
        for i in range(self.num_poolings):
            x, before_pooling = self.build_downsampling_block(x, num_filters)
            self.down[i] = {
                'output': x,
                'before_pooling': before_pooling,
                'num_filters': num_filters
            }
            num_filters *= 2
        x = keras.layers.Dropout(rate=self.dropout_rate)(x)

        # build the colvolutional block at the lowest stage
        x = self.build_lowest_convolution_block(x, num_filters)
        self.lowest_stage_output = x

        # build the upsampling stage
        self.up = dict()
        for i in reversed(range(self.num_poolings)):
            associated_down = self.down[i]
            x = self.build_upsampling_block(x, associated_down['before_pooling'], associated_down['num_filters'])
            self.up[i] = {
                'output': x,
                'num_filters': associated_down['num_filters']
            }
        self.outputs = x

    def add_output_layer(self):
        """
        adds an output layer to the outputs of the model so far. The new outputs will have the specified
        number of output dimensions.
        """
        self.outputs = self.conv2d_layer(self.outputs, num_filters=self.output_dimensions)

    def instanciate_model(self):
        return keras.Model(inputs=self.inputs, outputs=self.outputs, name=self.model_name)

    def build_model(self):
        self.build_model_core()
        self.add_output_layer()
        return self.instanciate_model()

class ColorNetBuilder(UNetBuilder):
    def __init__(self, input_shape, 
                    output_dimensions=ColorEncoder.vec_size,
                    initial_num_filters=64, 
                    num_poolings=4, 
                    kernel_size=3,
                    dropout_rate=0.4,
                    batch_normalization=True):
        super(ColorNetBuilder, self).__init__(
            input_shape=input_shape,
            output_dimensions=output_dimensions,
            initial_num_filters=initial_num_filters, 
            num_poolings=num_poolings, 
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            batch_normalization=batch_normalization
        )

    def calculate_output_shape(self, input_shape):
        assert input_shape[0] % 2 == 0
        assert input_shape[1] % 2 == 0
        return (int(input_shape[0]/2), int(input_shape[1]/2), self.outputs.shape[-1].value)

    def add_output_layer(self):
        self.outputs = self.conv2d_layer(self.up[1]['output'], num_filters=self.output_dimensions, activation='softmax')