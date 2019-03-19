#!/usr/bin/env python3
"""
LSTM
Author: Christian Lang <me@christianlang.io>
"""

import keras



class GAN(object):
    def __init__(self, img_shape, latent_dim, channels):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.channels = channels

        self.img_rows = self.img_shape[0]
        self.img_cols = self.img_shape[1]

        optimizer = keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
                loss = 'binary_crossentropy',
                optimizer = optimizer,
                metrics = ['accuracy'])

        self.generator = self.build_generator()

        inp = keras.layers.Input(shape = (self.latent_dim,))
        image = self.generator(inp)

        self.discriminator.trainable = False

        validity = self.discriminator(image)

        self.combined = keras.models.Model(inp, validity)
        self.combined.compile(
                loss = 'binary_crossentropy',
                optimizer = optimizer
                )

    def build_generator(self):

        model = keras.models.Sequential()

        x, y = (self.img_rows // 2) // 2, (self.img_cols // 2) // 2

        model.add(keras.layers.Dense(128 * x * y, activation="relu", input_dim=100))

        model.add(keras.layers.Reshape((x, y, 128)))

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

        latent = keras.layers.Input(shape = (self.latent_dim,))
        image = model(latent)

        return keras.models.Model(latent, image)
    
    def build_discriminator(self):

        model = keras.models.Sequential()

        model.add(keras.layers.convolutional.Conv2D(
            32,
            kernel_size = 3,
            strides = 2,
            input_shape = self.img_shape,
            padding = "same"))

        model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.2))

        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.convolutional.Conv2D(
            64,
            kernel_size = 3,
            strides = 2,
            padding = "same"))

        model.add(keras.layers.ZeroPadding2D(
            padding = ((0, 1), (0, 1))))

        model.add(keras.layers.BatchNormalization(momentum = 0.8))

        model.add(keras.layers.advanced_activations.LeakyReLU(alpha = 0.2))

        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.convolutional.Conv2D(
            128,
            kernel_size = 3,
            strides = 2,
            padding = "same"))

        model.add(keras.layers.BatchNormalization(momentum = 0.8))

        model.add(keras.layers.advanced_activations.LeakyReLU(alpha = 0.2))

        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.convolutional.Conv2D(
            256,
            kernel_size = 3,
            strides = 1,
            padding = "same"))

        model.add(keras.layers.BatchNormalization(momentum = 0.8))

        model.add(keras.layers.advanced_activations.LeakyReLU(alpha = 0.2))

        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(1, activation='sigmoid'))

        image = keras.layers.Input(shape = self.img_shape)
        validity = model(image)
        
        return keras.models.Model(image, validity)

    def generate(self, X):
        return self.generator.predict(X)

    def discriminate(self, X):
        return self.discriminator.predict(X)

    def train_generator(self, X, y):
        return self.combined.train_on_batch(X, y)
    
    def train_discriminator(self, X_true, y_true, X_false, y_false):

        true_loss = self.discriminator.train_on_batch(X_true, y_true)
        false_loss = self.discriminator.train_on_batch(X_false, y_false)

        return (true_loss, false_loss)
