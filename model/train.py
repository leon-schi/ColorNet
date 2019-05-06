import os
import tensorflow as tf
from tensorflow import keras

from .input import inputs
from .model import ColorNetBuilder
from .color_mapping import ColorEncoder

class Model:
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/cp.h5')

    def __init__(self):
        self.session = None
        self.iterator = None

        builder = ColorNetBuilder(ColorEncoder.vec_size,
            initial_num_filters=16, 
            num_poolings=2,
        )
        self.model = builder.build_model()

        # restore weights from checkpoint if there is one
        try:
            self.model.load_weights(self.checkpoint_dir)
        except OSError:
            pass

        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_crossentropy'])

    def get_session(self):
        if self.session == None:
            self.session = tf.Session()
        return self.session

    def get_iterator(self):
        if self.iterator == None:
            self.iterator = inputs().make_one_shot_iterator()
        return self.iterator

    def train(self):
        checkpoint_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_dir, 
                                                save_weights_only=True,
                                                verbose=1)

        iterator = inputs().make_one_shot_iterator()

        sess = self.get_session() 
        bw, labels = sess.run(self.get_iterator().get_next())
        self.model.fit(x=bw, y=labels, epochs=2, callbacks=[checkpoint_callback])

def train_model():
    model = Model()
    model.train()