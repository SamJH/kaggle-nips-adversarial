from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

import argparse
import keras
from keras.models import load_model

from utils import load_images, custom_object_scope, ensemble
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', type=str, required=True, help='Input directory with images.')
  parser.add_argument('--output_file', type=str, required=True, help='Output file to save labels.')
  args = parser.parse_args()

  batch_size = 2
  image_height = 299
  image_width = 299
  nb_channels = 3
  
  batch_shape = [100, image_height, image_width, nb_channels]
  num_classes = 1000
  
  with custom_object_scope():
    adv_inception_resnet_v2 = load_model('models/adv_inception_resnet_v2.model', compile=False)
    adv_inception_v3 = load_model('models/adv_inception_v3.model', compile=False)

    inception_resnet_v2 = load_model('models/inception_resnet_v2.model', compile=False)
    inception_v3 = load_model('models/inception_v3.model', compile=False)
    resnet50 = load_model('models/resnet50.model', compile=False)
    vgg16 = load_model('models/vgg16.model', compile=False)
    vgg19 = load_model('models/vgg19.model', compile=False)
    xception = load_model('models/xception.model', compile=False)
  
  white_box_models = [adv_inception_resnet_v2, 
                      adv_inception_v3,
                      inception_resnet_v2,
                      inception_v3,
                      vgg16,
                      vgg19,
                      xception,
                      resnet50
                     ]
  
  voting_model = ensemble(white_box_models, logits=True)
  
  with open(args.output_file, 'w') as f:
      for filenames, images in load_images(args.input_dir, batch_shape):
        logits = voting_model.predict(images, batch_size=batch_size, verbose=1)
        labels = np.argmax(logits, axis=-1) + 1


        for filename, label in zip(filenames, labels):
            f.write('{},{}\n'.format(filename, label))
