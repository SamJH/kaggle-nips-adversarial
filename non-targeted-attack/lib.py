from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
from glob import glob
from PIL import Image
from pandas import read_csv

from keras import backend as K
from keras.layers import Layer, Input, Add, Multiply
from keras.models import Model
from keras.utils import CustomObjectScope


class OneHot(Layer):
    def call(self, input):
        preds_max = K.max(input, axis=1, keepdims=True)
        y = K.cast(K.equal(input, preds_max), K.floatx())
        y = y / K.sum(y, axis=1, keepdims=True)
        return y


class RandomReset(Layer):
    def __init__(self, eps, clip_min=0., clip_max=255., **kwargs):
        super(RandomReset, self).__init__(**kwargs)
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def call(self, input):
        x = input + K.random_uniform(K.shape(input), -self.eps, +self.eps)
        return K.clip(x, self.clip_min, self.clip_max)


class PGD(Layer):
    def __init__(self, eps=16.0, eps_iter=7.0,
                 clip_min=0.0, clip_max=255.0,
                 from_logits=False,
                 sign=True,
                 loss='categorical_crossentropy',
                 **kwargs):
        self.eps = eps
        self.eps_iter = eps_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.from_logits = from_logits
        self.sign = sign
        self.loss = loss
        super(PGD, self).__init__(**kwargs)

    def call(self, input):
        x, y, x0, y0 = input

        if self.loss == 'categorical_crossentropy':
            loss = -K.categorical_crossentropy(y0, y, from_logits=self.from_logits)
        elif self.loss == 'max':
            loss = y0 * y
        elif self.loss == 'minmax':
            loss = K.max(y0 * y, axis=-1) - K.max((1 - y0) * y, axis=-1)

        loss = K.mean(loss)

        grad, = K.gradients(loss, [x])

        if self.sign:
            grad_sign = K.sign(grad)
        else:
            grad_sign = grad / K.mean(grad, axis=-1, keepdims=True)

        adv_x = x + self.eps_iter * grad_sign

        adv_x = x0 + K.clip(adv_x - x0, -self.eps, +self.eps)
        adv_x = K.clip(adv_x, self.clip_min, self.clip_max)
        adv_x = K.stop_gradient(adv_x)
        return adv_x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {'eps': self.eps,
                  'eps_iter': self.eps_iter,
                  'clip_min': self.clip_min,
                  'clip_max': self.clip_max,
                  'from_logits': self.from_logits
                  }
        base_config = super(PGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Crop(Layer):
    def __init__(self, height=224, width=224, position=None, **kwargs):
        self.height = height
        self.width = width
        if position is None:
            position = 'center'
        self.position = position
        super(Crop, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        super(Crop, self).build(input_shape)

    def call(self, input):
        if self.position == 'center':
            startY = (self.input_height - self.height) // 2
            endY = startY + self.height
            startX = (self.input_width - self.width) // 2
            endX = startX + self.width
        elif self.position == 'corner':
            startY = 0
            endY = self.height
            startX = 0
            endX = self.width
        else:
            raise NotImplementedError("not implemented yet!")

        return input[:, startY:endY, startX:endX]

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[1] = self.height
        input_shape[2] = self.width
        return tuple(input_shape)

    def get_config(self):
        config = {'height': self.height,
                  'width': self.width,
                  'position': self.position}
        base_config = super(Crop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Preprocessing(Layer):
    def __init__(self, format='imagenet_old', **kwargs):
        self.format = format
        super(Preprocessing, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Preprocessing, self).build(input_shape)

    def _imagenet_old(self, x, data_format=None):
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}

        mean = np.array([103.939,
                         116.779,
                         123.68], dtype=K.floatx())

        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            x = x[..., ::-1, :, :]
            # Zero-center by mean pixel
            x -= mean[..., np.newaxis, np.newaxis]
        else:
            # 'RGB'->'BGR'
            x = x[..., :, :, ::-1]
            # Zero-center by mean pixel
            x -= mean[...]
        return x

    def _imagenet_new(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def call(self, input):
        if self.format == 'imagenet_old':
            return self._imagenet_old(input)
        elif self.format == 'imagenet_new':
            return self._imagenet_new(input)
        else:
            raise NotImplementedError("not implemented yet!")

    def get_config(self):
        config = {'format': self.format}
        base_config = super(Preprocessing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def load_dataset(input_dir, metadata_csv=None, limit=100):
    search_pattern = os.path.join(input_dir, '*.png')
    filepaths = glob(search_pattern)
    filepaths = filepaths[:limit]
    filepaths.sort()

    def process_image(path):
        image = Image.open(path)
        image = np.array(image).astype(np.float32)
        return image

    images = np.stack([process_image(path) for path in filepaths])

    if metadata_csv:
        metadata = read_csv(metadata_csv)

        imageId = metadata['ImageId'].tolist()

        pattern = re.compile('([0-9a-f]+).png')
        hashes = [pattern.search(path).group(1)
                  for path in filepaths]

        trueLabel = metadata['TrueLabel'].tolist()
        trueLabel = dict(zip(imageId, trueLabel))
        trueLabel = np.stack(trueLabel[i] for i in hashes)

        targetClass = metadata['TargetClass'].tolist()
        targetClass = dict(zip(imageId, targetClass))
        targetClass = np.stack(targetClass[i] for i in hashes)

        return filepaths, images, trueLabel, targetClass
    else:
        return filepaths, images


def ensemble(models, logits=False):
    if len(models) == 1:
        return models[0]

    input_shape = models[0].input_shape
    for model in models:
        assert (input_shape == model.input_shape)

    x = Input(input_shape[1:])
    y = [model(x) for model in models]
    if logits:
        y = Add()(y)
    else:
        y = Multiply()(y)
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
