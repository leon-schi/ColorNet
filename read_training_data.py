import tensorflow as tf

from model.color_mapping import ColorEncoder
from model.input import inputs

tf.enable_eager_execution()

if __name__ == "__main__":
    dataset = inputs()
    encoder = ColorEncoder()

    for bw, labels in dataset:
        bw, labels = bw.numpy(), labels.numpy()
    
        encoder.decode_and_show(labels, bw)