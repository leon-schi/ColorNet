import unittest
import numpy as np

from tensorflow import keras

from model.model import ColorNetBuilder

class ModelTest(unittest.TestCase):

    def setUp(self):
        self.builder = ColorNetBuilder(400)
        self.model = self.builder.build_model()

    def testSummary(self):
        print(self.model.summary())

    def testCompilation(self):
        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    def testPrediction(self):
        data = np.random.uniform(size=(1, 256, 256, 1))
        prediction = self.model.predict(data, batch_size=1, verbose=1)
        
        assert prediction.shape[-1] == self.builder.output_dimensions
        
if __name__ == "__main__":
    unittest.main()