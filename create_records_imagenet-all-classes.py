import requests as r
import numpy as np
import pickle
import io

import tensorflow as tf
from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

NUM_PICS = 20000
BATCH_SIZE = 1024

img_size = (256, 256)

def fetch_and_store(url, writer):
    try:
        q = r.get(url, headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}, timeout=5)
        if q.status_code == 200:
            if 'photo_unavailable' in q.url: 
                print('     FAILED!')
                return False

            img = Image.open(io.BytesIO(q.content))
            if np.mean(np.array(img.convert('HSV')).T[1]) / 255 < 0.05:
                print('     FAILED! Black White Image Detected')
                return False

            img = img.convert('YCbCr')
            h, w = img.size
            s = min(h, w)
            top, bottom = (h-s)/2, (w-s)/2
            img = img.crop((top, bottom, top + s, bottom + s))
            img = img.resize(img_size, Image.ANTIALIAS)

            img = np.array(img)
            img = img / 255
            img = img.astype(np.float32)
            lum = img.T[0].T
            color = np.array([img.T[1].T, img.T[2].T])

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(img.shape[0]),
                'width': _int64_feature(img.shape[1]),
                'bw': _bytes_feature(lum.tobytes()),
                'color': _bytes_feature(color.tobytes())}))
            
            writer.write(example.SerializeToString())
            return True
        print('     FAILED!')
        return False
    except:
        print('     FAILED!')
    return False

def read_url(f):
    try:
        line = f.readline()
        if not line:
            return None
        url = line.split('\t')[1]
        return url
    except: return None

if __name__ == "__main__":
    f = open('imagenet-urls/fall11_urls.txt', 'r')
    num_pics_saved = 0
    num_pics_in_current_file = 0
    num_file = 20
    i = 1

    writer = tf.io.TFRecordWriter('./imagenet-records-all-classes/records-' + str(num_file))
    while True:
        if num_pics_in_current_file >= BATCH_SIZE:
            writer.close()
            num_file += 1
            num_pics_in_current_file = 0
            writer = tf.io.TFRecordWriter('./imagenet-records-all-classes/records-' + str(num_file))

        url = read_url(f)
        print('[ fetching image %i | %i of %i in current file (%i) ] %s' % (num_pics_saved, num_pics_in_current_file, BATCH_SIZE, num_file, url))
        if fetch_and_store(url, writer):
            num_pics_saved += 1
            num_pics_in_current_file += 1
        i += 1
    writer.close()