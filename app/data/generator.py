#!/usr/bin/env python3

import os

import numpy as np


class Generator(object):
    def __init__(self, dims, steps):
        self.frames = len(os.listdir('tmp/frames'))
        self.dims = dims
        self.steps = steps
        pass

    def __call__(self, target_frame = None):

        if target_frame is None:
            target_frame = np.random.choice(self.frames)

        X = []
        for i in range(self.steps, 0, -1):
            if target_frame - i > 0:
                X.append(np.load('tmp/frames/{}.npy'.format(target_frame - i)).flatten())

            else:
                X.append(np.zeros(self.dims))

        X = np.array(X) / 255.0
        y = np.load('tmp/frames/{}.npy'.format(target_frame)).flatten() / 255.0

        return (X, y)
