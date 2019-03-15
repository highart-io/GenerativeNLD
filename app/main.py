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
from .models.lstm import LSTM


steps = 4
batch_size = 32

def main():

    if (len(sys.argv) > 1) and (sys.argv[1].startswith('y')):
        init()

    dims = np.load('tmp/frames/0.npy').shape
    input_shape = (steps, dims[0] * dims[1])
    output_dim = dims[0] * dims[1]
    
    model = LSTM(input_shape, output_dim)
    generator = Generator(dims = dims, steps = steps)

    iteration = 0
    while True:
        
        batch = [generator() for i in range(batch_size)]
        X, y = np.array([x[0] for x in batch]), np.array([y[1] for y in batch])

        loss = model.train_on_batch(X, y)

        iteration += 1

        print('Iteration : {} | Loss : {}'.format(iteration, loss))

    return
if __name__ == '__main__':
    main()
