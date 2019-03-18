#!/usr/bin/env python3
"""
Main
Author: Christian Lang <me@christianlang.io>
"""
import sys

import datetime as dt
import numpy as np

from .data.generator import Generator
from .utils.init import init
from .utils.make_gif import make_gif
from .models.lstm import LSTM


steps = 12
batch_size = 32
gif_cadence = 10

def main():

    if (len(sys.argv) > 1) and (sys.argv[1].startswith('y')):
        init()

    dims = np.load('tmp/frames/0.npy').shape
    input_shape = (steps, dims[0] * dims[1])
    output_dim = dims[0] * dims[1]
    
    model = LSTM(input_shape, output_dim)
    generator = Generator(dims = dims, steps = steps)
    
    make_gif(name = 'epoch_0', starting_frame = 10000, frames = 60, dims = dims, model = model, generator = generator, fps = 6)
    
    epochs = 0
    while epochs < 1000:
        
        model.fit_generator(
                generator = generator,
                use_multiprocessing = True
                )

        epochs += 1
        if epochs % gif_cadence == 0:
            make_gif(name = 'epoch_{}'.format(epochs), starting_frame = 10000, frames = 60, dims = dims, model = model, generator = generator, fps = 6)

    exit(0)

    return
if __name__ == '__main__':
    main()
