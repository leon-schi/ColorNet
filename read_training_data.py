import tensorflow as tf

from model.color_mapping import ColorEncoder
from model.input import inputs

tf.enable_eager_execution()

if __name__ == "__main__":
    dataset = inputs()
    encoder = ColorEncoder()

    for bw_batch, labels_batch in dataset:
        #bw_batch, labels_batch = bw.numpy(), labels.numpy()

        print(bw_batch.numpy().shape)

        for bw, labels in zip(bw_batch.numpy(), labels_batch.numpy()):
            encoder.decode_and_show(labels, bw)