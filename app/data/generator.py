#!/usr/bin/env python3

import os

import numpy as np


class Generator(object):
    def __init__(self, dims):
        self.frames = len(os.listdir('tmp/frames'))
        self.dims = dims
        pass

    def __call__(self, steps = 4):

        target_frame = np.random.choice(self.frames)

        X = []
        for i in range(steps, 0, -1):
            if target_frame - i > 0:
                X.append(np.load('tmp/frames/{}.npy'.format(target_frame - i)))

        X = np.array(X)
        y = np.load('tmp/frames/{}.npy'.format(target_frame))

        return (X, y)
