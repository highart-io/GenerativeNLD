#!/usr/bin/env python3

import numpy as np
import imageio


def make_gif(name, starting_frame, frames, dims, model, generator, fps = 30):

    images = []

    X, _ = generator(starting_frame)

    for x in X:
        images.append(np.round(x * 255).reshape(dims))

    while len(images) < frames:

        data = X.reshape(1, X.shape[0], X.shape[1])

        y = model.predict(data)

        images.append(np.round(y * 255).reshape(dims))

        X = np.append(X, y, axis = 0)[1:]

    imageio.mimwrite(
            'gifs/{}.gif'.format(name),
            [image.astype(np.uint8) for image in images],
            fps = fps)

    return True
