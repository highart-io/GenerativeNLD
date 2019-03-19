#!/usr/bin/env python3
"""
LSTM
Author: Christian Lang <me@christianlang.io>
"""

import keras


model = keras.models.Sequential()

model.add(keras.layers.Dense(128 * 7 * 7, activation="relu", input_dim=100))
model.add(keras.layers.Reshape((7, 7, 128)))
model.add(keras.layers.convolutional.UpSampling2D())
model.add(keras.layers.convolutional.Conv2D(128, kernel_size=4, padding="same"))
model.add(keras.layers.BatchNormalization(momentum=0.8))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.convolutional.UpSampling2D())
model.add(keras.layers.convolutional.Conv2D(64, kernel_size=3, padding = "same"))
model.add(keras.layers.BatchNormalization(momentum=0.8))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.convolutional.Conv2D(1, kernel_size = 3, padding = "same"))
model.add(keras.layers.Activation("tanh"))

model.summary()

class GAN(object):
    def __init__(self, img_shape, latent_dim, channels):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.channels = channels
