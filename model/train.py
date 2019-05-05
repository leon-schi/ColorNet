import os
import tensorflow as tf
from tensorflow import keras

from .input import inputs
from .model import ColorNetBuilder
from .color_mapping import ColorEncoder

class Model:
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

    def __init__(self):
        builder = ColorNetBuilder(ColorEncoder.vec_size)
        self.model = builder.build_model()

        try:
            self.model.load_weights(self.checkpoint_dir)
        except OSError:
            pass

        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_crossentropy'])

    def train(self):
        checkpoint_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_dir, 
                                                save_weights_only=False,
                                                verbose=1)

        iterator = inputs().make_one_shot_iterator()

        with tf.Session() as sess:
            bw, labels = sess.run(iterator.get_next())
            self.model.fit(x=bw, y=labels, epochs=2, callbacks=[checkpoint_callback])

def train_model():
    model = Model()
    model.train()