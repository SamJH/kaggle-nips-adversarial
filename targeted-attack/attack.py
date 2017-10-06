from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import argparse
import numpy as np
from time import time
from PIL import Image
from tqdm import tqdm

import keras
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils import np_utils

from lib import load_dataset
from lib import PGD
from lib import custom_object_scope

z = time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',   type=str,   required=True)
    parser.add_argument('--output_dir',  type=str,   required=True)
    parser.add_argument('--nb_classes',  type=int,   default=1000)
    parser.add_argument('--width',       type=int,   default=299)
    parser.add_argument('--height',      type=int,   default=299)
    parser.add_argument('--nb_channels', type=int,   default=3)
    parser.add_argument('--nb_iter',     type=int,   default=6)
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--max_eps',     type=float, default=16.0)
    parser.add_argument('--eps_iter',    type=float, default=6.0)
    parser.add_argument('--clip_min',    type=float, default=0.0)
    parser.add_argument('--clip_max',    type=float, default=255.0)
    parser.add_argument('--loss',        type=str,   default='minmax')
    parser.add_argument('--nosign',      action='store_true')
    args = parser.parse_args()
    
    args.eps_iter = args.max_eps // 3
    
    paths, images = load_dataset(args.input_dir, limit=100)
    
    with open(os.path.join(args.input_dir, 'target_class.csv'), 'r') as f:
        targets = {row[0]: int(row[1])
                   for row in csv.reader(f)
                   if len(row) >= 2}

    targets = np.array([targets[i.split('/') [-1]] for i in paths])
    targets = (targets + args.nb_classes - 1) % args.nb_classes
    targets = np_utils.to_categorical(targets, args.nb_classes)
    
    print('Loading model...', time()-z)
    with custom_object_scope():
        voting_model = load_model('VOTING.model')
    print('Done!', time()-z)
        
        
    def build_attacker(white_box_model):
        attack = PGD(eps=args.max_eps,
                     eps_iter=args.eps_iter,
                     clip_min=args.clip_min,
                     clip_max=args.clip_max,
                     from_logits=True,
                     sign=not args.nosign,
                     loss=args.loss)

        x = Input(shape=(args.height, args.width, args.nb_channels))
        y = white_box_model(x)
        
        x0 = Input(shape=(args.height, args.width, args.nb_channels))
        y0 = Input(shape=(args.nb_classes,))

        adv_x = attack([x, y, x0, y0])
        return Model(inputs=[x, x0, y0],
                     outputs=adv_x,
                     name='PGD_adv')

    attacker = build_attacker(voting_model)

    X = images
    Y = targets

    def craft(attacker, X):
        start_X = X + np.random.uniform(-args.max_eps, +args.max_eps, size=X.shape)
        start_X = np.clip(start_X, args.clip_min, args.clip_max)
        adv_x = [ start_X  ] + [ None ] * args.nb_iter
        
        last_adv_X = start_X.copy()
        for i in range(args.nb_iter):
            for j in range(0, 100, 5):
                last_adv_X[j:j+5] = attacker.predict([last_adv_X[j:j+5],
                                                      X[j:j+5],
                                                      Y[j:j+5]],
                                                     batch_size=5,
                                                     verbose=1)
            
                if time() > z + 500 - 30:
                    print('breaking...', i + 1, j)
                    return last_adv_X

        return last_adv_X
    
    adv_X = craft(attacker, X)
    
    for path, img in tqdm(zip(paths, adv_X)):
        path = path.split('/') [-1]
        path = os.path.join(args.output_dir, path)
        Image.fromarray(img.astype(np.uint8)).save(path)
