import numpy as np
from scipy.ndimage import zoom
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf

def gaussian(x, sig):
    return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))

class ColorEncoder:
    """
    A class that can transform an colorized image to a vector, which is containing probabilities
    for a certain number of color calasses, for each pixel and vice-versa. This encoded vecors are used
    for training the model
    """
    
    GRID_SIZE = (20, 20) # the size of the grid in color space

    # make an array containing the a,b values for each grid field
    a = np.linspace(0, 1, num=GRID_SIZE[0], endpoint=False) + 1/(2*GRID_SIZE[0])
    b = np.linspace(0, 1, num=GRID_SIZE[1], endpoint=False) + 1/(2*GRID_SIZE[1])
    ab_lookup = np.transpose([np.tile(a, len(b)), np.repeat(b, len(a))])

    input_size = 256
    output_size = 128
    vec_size = len(ab_lookup)

    nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(ab_lookup)

    def __init__(self, dtype=tf.float32):
        self.dtype = dtype

    def encode_sparse(self, color_channels):
        assert self.input_size == len(color_channels[0])
        # sample down to output size
        a = zoom(color_channels[0], self.output_size/self.input_size)
        b = zoom(color_channels[1], self.output_size/self.input_size)
        color_channels = np.array([a, b])

        dense_shape = (self.output_size, self.output_size, self.vec_size)
        indices, values = [], []
        for y in range(self.output_size):
            for x in range(self.output_size):
                # get actual a,b values
                ab = np.array([[color_channels[0][y, x], color_channels[1][y, x]]])
                # find the 5 closest a,b values from the grid
                distances, ab_indices = self.nbrs.kneighbors(ab)
                vector = np.zeros((self.vec_size))
                # find weights for each of the 5 closest points
                weights = [gaussian(d, 3) for d in distances[0]]
                sum_weights = sum(weights)
                
                indices += [[x, y, i] for i in ab_indices[0]]
                values += [w / sum_weights for w in weights]
        labels = tf.SparseTensor(
            indices=tf.constant(indices, dtype=tf.int64), 
            values=tf.constant(values, dtype=self.dtype), 
            dense_shape=tf.constant(dense_shape, dtype=tf.int64))
        return tf.sparse_reorder(labels)

    def encode(self, color_channels):
        assert tf.executing_eagerly()
        sparse_labels = self.encode_sparse(color_channels)
        return tf.sparse_tensor_to_dense(sparse_labels).numpy()

    def decode(self, labels):
        assert labels.shape == (self.output_size, self.output_size, self.vec_size)

        a, b = np.zeros((self.output_size, self.output_size)), np.zeros((self.output_size, self.output_size))
        for y in range(self.output_size):
            for x in range(self.output_size):
                # use the a, b value with the highest probability
                index = np.argmax(labels[x, y])
                value = self.ab_lookup[index]
                a[x, y] = value[0]
                b[x, y] = value[1]

        # upsampling
        a = zoom(a, self.input_size/self.output_size)
        b = zoom(b, self.input_size/self.output_size)
        color_channels = np.array([a, b])
        return color_channels

    def decode_and_show(self, labels, black_white):
        from PIL import Image
        import matplotlib.pyplot as plt

        color_channels = self.decode(labels)
        black_white = black_white.reshape((self.input_size, self.input_size))
        img = 255*np.array([black_white, color_channels[0], color_channels[1]]).T
        img = img.astype(np.uint8)

        o = Image.fromarray(img, mode='YCbCr')
        o.show()
        
        plt.figure()
        plt.imshow(black_white.T, cmap='gist_gray')
        plt.show()