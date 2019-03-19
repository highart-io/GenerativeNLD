#!/usr/bin/env python3

import os

import numpy as np
import keras


class Generator(keras.utils.Sequence):
    def __init__(self):
        self.frames = np.arange(len(os.listdir('tmp/frames')))
        pass

    def __call__(self, target_frame = None):

        if target_frame is None:
            target_frame = np.random.choice(self.frames)

        X = np.load('tmp/frames/{}.npy'.format(target_frame))

        X = np.array(X) / 255.0

        return X
