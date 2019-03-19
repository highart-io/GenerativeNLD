#!/usr/bin/env python3

import os

import numpy as np
import keras


class Generator(keras.utils.Sequence):
    def __init__(self, channels):
        self.frames = np.arange(len(os.listdir('tmp/frames')))
        self.channels = channels
        pass

    def __call__(self, target_frame = None):

        if target_frame is None:
            target_frame = np.random.choice(self.frames)

        X = np.load('tmp/frames/{}.npy'.format(target_frame))
        X = X.reshape(X.shape[0], X.shape[1], self.channels)

        X = np.array(X) / 127.5 - 1

        return X
