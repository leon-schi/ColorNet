import os
import re
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
    checkpoint_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    checkpoint_dir = os.path.join(checkpoint_folder, 'cp.{epoch}.h5')
    epoch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/epoch')
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    def __init__(self):
        self.sess = None
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
            input_shape=(None, None, 1),#model_architecture['input_shape'],
            initial_num_filters=model_architecture['num_filters'],
            num_poolings=model_architecture['num_poolings'],
        )
        self.model = builder.build_model()

        input_shape = model_architecture['input_shape']
        output_shape = builder.calculate_output_shape(input_shape)
        print(input_shape, output_shape)
        self.input_provider = InputProvider(CONFIG['input_config'], input_shape, output_shape)
        #self.input_provider = InputProvider(CONFIG['input_config'], builder.input_shape, builder.output_shape)
        self.inputs = self.input_provider.inputs()
        self.iterator = self.inputs.make_one_shot_iterator().get_next()

    def compile_model(self):
        # load the epoch, when training was stopped
        try:
            loaded = joblib.load(self.epoch_dir)
            self.initial_epoch = loaded['epoch']
            learning_rate = loaded['lr']
        except OSError:
            self.initial_epoch = 0
            learning_rate = self.learning_rate

        print('training will start at epoch {} with learning rate {}'.format(self.initial_epoch, learning_rate))

        self.model.compile(optimizer=keras.optimizers.Adam(
            lr=learning_rate,
            decay=self.learning_rate_decay),
            loss='categorical_crossentropy',
            metrics=['categorical_crossentropy'])

        # restore weights from checkpoint if there is one
        try:
            checkpoint_file = sorted([f for f in os.listdir(self.checkpoint_folder) if re.match('cp.*.h5', f)])[-1]
            self.model.load_weights(os.path.join(self.checkpoint_folder, checkpoint_file))
            print('loaded weights from file: {}'.format(checkpoint_file))
        except OSError:
            pass

    def save(self):
        self.model.save(os.path.join(self.checkpoint_folder, 'saved.h5'))

    def build_callbacks(self):
        checkpoint_callback = keras.callbacks.ModelCheckpoint(self.checkpoint_dir, 
                                                save_weights_only=False,
                                                verbose=0)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                update_freq=1)
        learning_rate_plateau = keras.callbacks.ReduceLROnPlateau(monitor='loss' ,factor=0.5, patience=3, min_delta=0.001, verbose=1)
        learning_rate_scheduler = keras.callbacks.LearningRateScheduler(
            schedule=lambda epoch, lr: lr * (1 - self.learning_rate_decay),
            verbose=1)

        def save_epoch(epoch, logs):
            joblib.dump({
                    'epoch': epoch + 1,
                    'lr': keras.backend.eval(self.model.optimizer.lr)
                }, self.epoch_dir)
            self.initial_epoch = epoch + 1
        save_epoch_callback = keras.callbacks.LambdaCallback(on_epoch_begin=save_epoch)

        self.callbacks = [
            checkpoint_callback,
            tensorboard_callback,
            learning_rate_scheduler,
            learning_rate_plateau,
            save_epoch_callback
        ]

    def train(self):
        self.model.fit(x=self.inputs,
                    initial_epoch=self.initial_epoch,
                    steps_per_epoch=self.steps_per_epoch,
                    epochs=self.num_epochs,
                    callbacks=self.callbacks)

    def get_session(self):
        if self.sess == None:
            self.sess = tf.Session()
        return self.sess

    def predict(self):
        sess = self.get_session()
        bws, _ = sess.run(self.iterator)
        labels = self.model.predict(bws)
        for bw, label in zip(bws, labels):
            ColorEncoder().decode_and_show(label, bw)