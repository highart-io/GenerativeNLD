#!/usr/bin/env python3
"""
Init
Author: Christian Lang <me@christianlang.io>
"""

import os
import shutil

import numpy as np
import cv2
import requests


def init():

    if not os.path.exists('tmp/'):
        print('Creating Directory')
        os.mkdir('tmp/')
        os.mkdir('tmp/frames')
        os.mkdir('gifs')

    else:
        print('Deleting Directory')
        shutil.rmtree('tmp/')
        os.mkdir('tmp/')
        os.mkdir('tmp/frames')
        os.mkdir('gifs')

    with open('sources/films.txt', 'r') as films:

        lines = films.read().splitlines()
        for line in lines:

            print('Downloading : {}'.format(line.split('/')[-1]))
            res = requests.get(line)

            with open('tmp/{}'.format(line.split('/')[-1]), 'wb') as f:
                print('Writing : {}'.format(line.split('/')[-1]))
                f.write(res.content)

    files = os.listdir('tmp/')

    vid = cv2.VideoCapture('tmp/{}'.format(files[0]))

    frame = 0
    while vid.isOpened():

        success, image = vid.read()
        if success:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            np.save('tmp/frames/{}.npy'.format(frame), image)

            frame += 1

        else:
            break

    vid.release()

    return
