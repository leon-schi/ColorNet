import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow import keras

from .config import CONFIG
from .input import inputs
from .model import ColorNetBuilder
from .color_mapping import ColorEncoder

class Model:
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/cp.h5')
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    def __init__(self):
        self.session = None
        self.iterator = None
        self.load_config()
        self.build_model()

    def load_config(self):
        training_config = CONFIG['training_config']
        self.num_epochs = training_config['num_epochs']
        self.learning_rate = training_config['learning_rate']
        self.learning_rate_decay = training_config['learning_rate_decay']

    def build_model(self):
        model_architecture = CONFIG['model_architecture']
        builder = ColorNetBuilder(
            initial_num_filters=model_architecture['num_filters'],
            num_poolings=model_architecture['num_poolings'],
        )
        self.model = builder.build_model()

        # restore weights from checkpoint if there is one
        try:
            self.model.load_weights(self.checkpoint_dir)
        except OSError:
            pass

        self.model.compile(optimizer=keras.optimizers.Adam(
            lr=self.learning_rate, 
            decay=self.learning_rate_decay),
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
                                                verbose=0)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_dir)

        sess = self.get_session()
        bw, labels = sess.run(self.get_iterator().get_next())
        self.model.fit(x=bw, y=labels, 
                epochs=self.num_epochs,
                callbacks=[checkpoint_callback, tensorboard_callback])

    def predict(self, images):
        if len(images.shape) < 4:
            images = np.array([images])

        labels = self.model.predict(images)
        ColorEncoder().decode_and_show(labels[0], images[0])