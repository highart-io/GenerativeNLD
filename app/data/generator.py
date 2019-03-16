#!/usr/bin/env python3

import os

import numpy as np
import keras


class Generator(keras.utils.Sequence):
    def __init__(self, dims, steps = 4, batch_size = 32, shuffle = True):
        self.frames = np.arange(len(os.listdir('tmp/frames')))
        self.dims = dims
        self.steps = steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        pass

    def __len__(self):
        return int(np.floor(len(self.frames) / self.batch_size))

    def __getitem__(self, index):

        frames = self.frames[index*self.batch_size:(index + 1) * self.batch_size]

        batch = [self.__call__(target_frame = frame) for frame in frames]

        X, y = np.array([x[0] for x in batch]), np.array([y[1] for y in batch])

        return (X, y)

    def __call__(self, target_frame = None):

        if target_frame is None:
            target_frame = np.random.choice(self.frames)

        X = []
        for i in range(self.steps, 0, -1):
            if target_frame - i > 0:
                X.append(np.load('tmp/frames/{}.npy'.format(target_frame - i)).flatten())

            else:
                X.append(np.zeros(self.dims).flatten())

        X = np.array(X) / 255.0
        y = np.load('tmp/frames/{}.npy'.format(target_frame)).flatten() / 255.0

        return (X, y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.frames)
        return
