import argparse

import os
import sys
import cv2
import math 
import numpy as np
from model.train import Model
from model.color_mapping import ColorEncoder

parser = argparse.ArgumentParser(description='Colorize an image with ColorNet')

parser.add_argument('-image', metavar='image', type=str, help='the path of the image to use')

from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(dir_path, '')

if __name__ == "__main__":
    args = parser.parse_args()
    
    image_name = args.image
    print(image_name)
    model = Model()
    
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape((1, image.shape[0], image.shape[1], 1)).T / 255

    labels = model.predict(image)
    encoder = ColorEncoder()
    img, label = image[0], labels[0]

    label = label[:math.floor(img.shape[0]/2), :math.floor(img.shape[1]/2), :]
    result = encoder.decode_to_final_image(label, img)

    img = Image.fromarray(result, mode='YCbCr')
    img.show()