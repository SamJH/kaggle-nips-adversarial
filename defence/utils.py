from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf

from PIL import Image

def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 1.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images



def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')
        


import numpy as np

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, InputLayer
from keras.layers import Add, Multiply
from keras.utils import CustomObjectScope

from layers import Crop, Preprocessing, PGD, OneHot


def ensemble(models, logits=False):
    if len(models) == 1:
        return models[0]
    
    input_shape = models[0].input_shape
    for model in models:
        assert(input_shape == model.input_shape)
    
    x = Input(input_shape[1:])
    y = [model(x) for model in models]
    if logits:
        y = Add() (y)
    else:
        y = Multiply() (y)
    return Model(inputs=x, outputs=y, name='ensemble')

def categorical_crossentropy_from_logits(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

def custom_object_scope():
    return CustomObjectScope({
        'categorical_crossentropy_from_logits': categorical_crossentropy_from_logits,
        'PGD': PGD,
        'Crop': Crop,
        'Preprocessing': Preprocessing,
        'OneHot': OneHot
    })
