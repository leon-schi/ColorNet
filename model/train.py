import os
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
from tensorflow import keras

from .config import CONFIG
from .input import InputProvider
from .model import ColorNetBuilder
from .color_mapping import ColorEncoder

class Model:
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/cp.h5')
    epoch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/epoch')
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    def __init__(self):
        self.load_config()
        self.build_model()
        self.compile_model()
        self.build_callbacks()

    def load_config(self):
        training_config = CONFIG['training_config']
        self.num_epochs = training_config['num_epochs']
        self.steps_per_epoch = training_config['steps_per_epoch']
        self.learning_rate = training_config['learning_rate']
        self.learning_rate_decay = training_config['learning_rate_decay']

    def build_model(self):
        model_architecture = CONFIG['model_architecture']
        builder = ColorNetBuilder(
            input_shape=model_architecture['input_shape'],
            initial_num_filters=model_architecture['num_filters'],
            num_poolings=model_architecture['num_poolings'],
        )
        self.model = builder.build_model()
        self.input_provider = InputProvider(CONFIG['input_config'], builder.input_shape, builder.output_shape)
        self.inputs = self.input_provider.inputs()
        self.iterator = self.inputs.make_one_shot_iterator().get_next()

    def compile_model(self):
        self.model.compile(optimizer=keras.optimizers.Adam(
            lr=self.learning_rate, 
            decay=self.learning_rate_decay),
              loss='categorical_crossentropy',
              metrics=['categorical_crossentropy'])

        # restore weights from checkpoint if there is one
        try:
            self.model.load_weights(self.checkpoint_dir)
            print('loaded')
        except OSError:
            pass

        # load the epoch, when training was stopped
        try:
            self.initial_epoch = joblib.load(self.epoch_dir)
        except OSError:
            self.initial_epoch = 0

    def build_callbacks(self):
        checkpoint_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_dir, 
                                                save_weights_only=False,
                                                verbose=0)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                update_freq=1)
        learning_rate_callback = keras.callbacks.ReduceLROnPlateau(monitor='loss' ,factor=0.5, patience=3, min_delta=0.001, verbose=1)

        def save_epoch(epoch, logs):
            joblib.dump(epoch, self.epoch_dir)
            self.initial_epoch = epoch
        save_epoch_callback = keras.callbacks.LambdaCallback(on_epoch_begin=save_epoch)

        self.callbacks = [
            checkpoint_callback,
            tensorboard_callback,
            learning_rate_callback,
            save_epoch_callback
        ]

    def train(self):
        self.model.fit(x=self.inputs,
                    initial_epoch=self.initial_epoch,
                    steps_per_epoch=self.steps_per_epoch,
                    epochs=self.num_epochs,
                    callbacks=self.callbacks)

    def predict(self):
        with tf.Session() as sess:
            bws, _ = sess.run(self.iterator)
        labels = self.model.predict(bws)
        for bw, label in zip(bws, labels):
            ColorEncoder().decode_and_show(label, bw)