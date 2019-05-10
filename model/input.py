import os
import numpy as np
import tensorflow as tf

from .color_mapping import ColorEncoder
from .config import CONFIG

dir_path = os.path.dirname(os.path.realpath(__file__))

input_config = CONFIG['input_config']
PREFETCH_BUFFER_SIZE = 64
SHUFFLE_BUFFER_SIZE = 128

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class Serializer:
    dtype = tf.float32
    encoder = ColorEncoder(dtype=dtype)

    def serialize_tensor(self, tensor):
        return tf.serialize_tensor(tensor).numpy()

    def serialize_sparse_tensor(self, sparse_tensor):
        return tf.serialize_tensor(tf.serialize_sparse(sparse_tensor)).numpy()

    def serialize_example(self, black_white, color_channels):
        assert tf.executing_eagerly()
        assert black_white.shape == (ColorEncoder.input_size, ColorEncoder.input_size, 1)

        label_tensor = self.encoder.encode_sparse(color_channels)
        black_white_tensor = tf.constant(black_white, dtype=self.dtype)
        color_channel_tensor = tf.constant(color_channels, dtype=self.dtype)
        
        return tf.train.Example(features=tf.train.Features(feature={
            'bw': _bytes_feature(self.serialize_tensor(black_white_tensor)),
            'color': _bytes_feature(self.serialize_tensor(color_channel_tensor)),
            'labels': _bytes_feature(self.serialize_sparse_tensor(label_tensor))
        }))

    @classmethod
    def parse_sparse_tensor(self, tensor):
        parsed = tf.parse_tensor(tensor, out_type=tf.string)
        sparse = tf.SparseTensor(
            indices=tf.parse_tensor(tf.gather(parsed, 0), out_type=tf.int64),
            values=tf.parse_tensor(tf.gather(parsed, 1), out_type=self.dtype),
            dense_shape=tf.parse_tensor(tf.gather(parsed, 2), out_type=tf.int64))
        return tf.sparse_tensor_to_dense(sparse)

    @classmethod
    def parse_example(self, example):
        feature_description = {
            'bw': tf.FixedLenFeature([], tf.string),
            'color': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string)
        }
        parsed = tf.parse_single_example(example, feature_description)

        black_white = tf.parse_tensor(parsed['bw'], self.dtype)
        labels = self.parse_sparse_tensor(parsed['labels'])        
        return black_white, labels

def parse(example):
    return Serializer.parse_example(example)

def test_inputs():
    dataset = tf.data.TFRecordDataset(os.path.join(dir_path, '../training-data/records-1'))
    dataset = dataset.map(map_func=parse)
    dataset = dataset.batch(batch_size=1)
    return dataset

def inputs():
    files = tf.data.Dataset.list_files(os.path.join(dir_path, '../training-data/*'))
    
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.map(map_func=parse, num_parallel_calls=input_config['num_parallel_map_calls'])
    dataset = dataset.batch(batch_size=input_config['batch_size'])
    dataset = dataset.repeat()
    return dataset

def inputs_pipeline():
    files = tf.data.Dataset.list_files(os.path.join(dir_path, '../training-data/*'))
    
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    dataset = dataset.map(map_func=parse, num_parallel_calls=input_config['num_parallel_map_calls'])
    dataset = dataset.batch(batch_size=input_config['batch_size'])
    dataset = dataset.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)
    return dataset