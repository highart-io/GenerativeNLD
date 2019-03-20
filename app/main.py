#!/usr/bin/env python3
"""
Main
Author: Christian Lang <me@christianlang.io> """
import sys

import datetime as dt
import numpy as np
import imageio
import os

from .data.generator import Generator
from .utils.init import init
from .models.gan import GAN
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

latent_dim = int(config['TRAINING']['LATENT_DIM'])
channels = int(config['TRAINING']['CHANNELS'])
batch_size = int(config['TRAINING']['BATCH_SIZE'])
inner_loop = int(config['TRAINING']['INNER_LOOP'])

def main():

    if (len(sys.argv) > 1) and (sys.argv[1].startswith('y')):
        init()

    img_shape = np.load('tmp/frames/0.npy').shape
    img_shape = (img_shape[0], img_shape[1], channels)
    
    gan = GAN(img_shape = img_shape, latent_dim = latent_dim, channels = channels)
    generator = Generator(channels = channels)

    y_true, y_false = np.ones((batch_size, 1)), np.zeros((batch_size, 1))

    iteration = 1
    while True:

        d_loss = []
        for _ in range(inner_loop):

            X_true = np.array([generator() for _ in range(batch_size)])

            noise = np.random.normal(loc = 0, scale = 1, size = (batch_size, latent_dim))
            X_false = gan.generate(X = noise)

            d_losses = gan.train_discriminator(X_true, y_true, X_false, y_false)
            d_l = np.add(d_losses[0], d_losses[1]) * 0.5

            d_loss.append(d_l)

        d_loss = np.array(d_loss).mean(axis = 0)
        
        noise = np.random.normal(loc = 0, scale = 1, size = (batch_size, latent_dim))

        g_loss = gan.train_generator(X = noise, y = y_true)

        print('Iteration : {} | Discriminator Loss : {} | Accuracy : {} | Generator Loss : {}'.format(iteration, d_loss[0], d_loss[1], g_loss))

        if (iteration % 10 == 0) or (iteration == 1):

            dirpath = 'imgs/iteration_{}/'.format(iteration)
            os.mkdir(dirpath)

            noise = np.random.normal(loc = 0, scale = 1, size = (10, latent_dim))
            gen_images = gan.generate(X = noise)

            gen_images = np.squeeze(np.round((gen_images + 1) * 127.5)).astype(np.uint8)

            for i in range(len(gen_images)):
                imageio.imwrite(
                        uri = dirpath + '{}.png'.format(i),
                        im = gen_images[i])
        
        iteration += 1
    
    return
if __name__ == '__main__':
    main()
