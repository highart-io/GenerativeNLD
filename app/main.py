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
from .models.gan import GAN


exit(0)

latent_dim = 100
channels = 1

def main():

    if (len(sys.argv) > 1) and (sys.argv[1].startswith('y')):
        init()

    img_shape = np.load('tmp/frames/0.npy').shape
    print(img_shape)
    
    gan = GAN(img_shape, latent_dim, channels)
    generator = Generator(dims = img_shape)

    exit(0)
    
    return
if __name__ == '__main__':
    main()
