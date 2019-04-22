import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

def read_binary(example, feature):
    return example.features.feature[feature].bytes_list.value[0]

def read_int64(example, feature):
    return example.features.feature[feature].int64_list.value[0]

if __name__ == "__main__":
    data = tf.data.TFRecordDataset('imagenet-records/records-0')
    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        while True:
            try:
                rec = sess.run(next_element)
            except tf.errors.OutOfRangeError:
                break

            example = tf.train.Example()
            example.ParseFromString(rec)

            bw = read_binary(example, 'bw')
            color = read_binary(example, 'color')
            height, width = read_int64(example, 'height'), read_int64(example, 'width')
            
            img_bw = np.fromstring(bw, dtype=np.float32).reshape((height, width))
            color_channels = np.fromstring(color, dtype=np.float32).reshape(2, height, width)
            
            img = 255*np.array([img_bw.T, color_channels[0].T, color_channels[1].T]).T
            img = img.astype(np.uint8)
            o = Image.fromarray(img, mode='YCbCr')
            o.show()
            
            plt.figure()
            plt.imshow(img_bw, cmap='gist_gray')
            plt.show()