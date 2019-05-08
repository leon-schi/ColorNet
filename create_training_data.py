import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

from model.input import Serializer

def read_binary(example, feature):
    return example.features.feature[feature].bytes_list.value[0]

def read_int64(example, feature):
    return example.features.feature[feature].int64_list.value[0]

tf.enable_eager_execution()

class RecordWriter:
    PATH = 'training-data/small'
    BATCH_SIZE = 256

    def current_filename(self):
        return 'records-' + str(self.num_file)

    def new_writer(self):
        self.num_file += 1
        self.num_pics_in_current_file = 0
        return tf.io.TFRecordWriter(os.path.join(self.PATH, self.current_filename()))

    def __init__(self):
        self.num_pics_saved = 0
        self.num_file = 0
        self.writer = self.new_writer()

    def write(self, example):
        if self.num_pics_in_current_file >= self.BATCH_SIZE:
            self.writer.close()
            self.writer = self.new_writer()

        self.writer.write(example.SerializeToString())
        self.num_pics_saved += 1
        self.num_pics_in_current_file += 1
        print('[saved image {}, {} in current file ({})]'
                .format(self.num_pics_saved, self.num_pics_in_current_file, self.num_file))

if __name__ == "__main__":
    files = tf.data.Dataset.list_files('imagenet-records/*')
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2)

    serializer = Serializer()
    writer = RecordWriter()

    for serialized in dataset:
        example = tf.train.Example()
        example.ParseFromString(serialized.numpy())

        bw = read_binary(example, 'bw')
        color = read_binary(example, 'color')
        height, width = read_int64(example, 'height'), read_int64(example, 'width')
        
        img_bw = np.frombuffer(bw, dtype=np.float32).reshape((1, height, width)).T
        color_channels = np.frombuffer(color, dtype=np.float32).reshape(2, height, width)
        
        example = serializer.serialize_example(img_bw, color_channels)
        writer.write(example)