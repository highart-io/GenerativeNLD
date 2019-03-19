#!/usr/bin/env python3
"""
Init
Author: Christian Lang <me@christianlang.io>
"""

import os
import shutil

import numpy as np
import imageio
import cv2
import requests
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

def init():

    scaling = int(config['DATA']['SCALING'])

    if not os.path.exists('tmp/') and not os.path.exists('imgs/'):
        print('Creating Directory')
        os.mkdir('tmp/')
        os.mkdir('imgs/')
        os.mkdir('tmp/films')
        os.mkdir('tmp/frames')

    else:
        print('Deleting Directory')
        if os.path.exists('tmp/'):
            shutil.rmtree('tmp/')
        if os.path.exists('imgs/'):
            shutil.rmtree('imgs/')
        os.mkdir('tmp/')
        os.mkdir('imgs/')
        os.mkdir('tmp/films')
        os.mkdir('tmp/frames')

    with open('sources/films.txt', 'r') as films:

        lines = films.read().splitlines()
        for line in lines:

            print('Downloading : {}'.format(line.split('/')[-1]))
            res = requests.get(line)

            with open('tmp/films/{}'.format(line.split('/')[-1]), 'wb') as f:
                print('Writing : {}'.format(line.split('/')[-1]))
                f.write(res.content)

    files = os.listdir('tmp/films/')

    with imageio.get_reader('tmp/films/{}'.format(files[0]), 'ffmpeg') as vid:

        frame = 0
        while True:

            try:
                image = vid.get_data(frame)
            except:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (image.shape[1] // scaling, image.shape[0] // scaling))

            np.save('tmp/frames/{}.npy'.format(frame), image)

            frame += 1
            if frame % 10000 == 0:
                print('Frame : {}'.format(frame))

    return
