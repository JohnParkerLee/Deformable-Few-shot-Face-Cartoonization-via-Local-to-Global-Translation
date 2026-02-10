#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    : detect.py
# @Time    : 2022/4/24 14:53
# @Author  : John
# @Software: PyCharm


import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import dlib


def get_landmark(filepath):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    paths = './weight/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(paths)
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm