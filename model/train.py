from tensorflow import keras

from .model import ColorNetBuilder
from .color_mapping import ColorEncoder

class Model:
    checkpoint_dir = 'checkpoints/model.hdf5'

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
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, 
                                                save_weights_only=True,
                                                verbose=1)