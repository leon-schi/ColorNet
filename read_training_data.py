import tensorflow as tf
tf.enable_eager_execution()

from model.color_mapping import ColorEncoder
from model.train import Model

if __name__ == "__main__":
    m = Model()
    encoder = ColorEncoder()

    for bw_batch, labels_batch in m.inputs:
        for bw, labels in zip(bw_batch.numpy(), labels_batch.numpy()):
            encoder.decode_and_show(labels, bw)