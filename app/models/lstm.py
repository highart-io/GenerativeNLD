#!/usr/bin/env python3
"""
LSTM
Author: Christian Lang <me@christianlang.io>
"""

import keras


def LSTM(input_shape, output_dim):

    img_input = keras.layers.Input(input_shape)

    x = keras.layers.CuDNNLSTM(
            units = 128,
            return_sequences = True)(img_input)

    x = keras.layers.Dropout(.5)(x)
    
    x = keras.layers.CuDNNLSTM(
            units = 128)(x)

    x = keras.layers.Dropout(.5)(x)

    target_img = keras.layers.Dense(
            units = output_dim,
            activation = 'sigmoid')(x)

    model = keras.models.Model(
            inputs = img_input,
            outputs = target_img)

    model.compile(
            optimizer = keras.optimizers.Adam(),
            loss = 'mean_squared_error')

    return model
