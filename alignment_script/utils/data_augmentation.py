#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import math
import numpy as np
import torchvision.transforms as transforms
import cv2
import copy

'''
# template
landAli = np.concatenate([landAli, np.ones((landAli.shape[0], 1))], axis=1)
landmarks_resize = np.dot(transfor.transform, landAli.T).T
'''

landmark_items = {'head': range(0, 17), 'L_eyebrows': range(17, 22), 'R_eyebrows': range(22, 27),
                  'eyebrows': range(17, 27), 'nose': range(27, 36), 'L_eyes': range(36, 42),
                  'R_eyes': range(42, 48), 'eyes': range(36, 48), 'mouth': range(48, 68)}

landmark_items_complete = {'head': range(0, 17),'head_All':range(0, 32), 'L_eyebrows': range(32, 37), 'R_eyebrows': range(37, 42),
                            'eyebrows': range(32, 42), 'nose': range(42, 51), 'L_eyes': range(51, 57),
                            'R_eyes': range(57, 63), 'eyes': range(51, 63), 'mouth': range(63, 83)}


def concatRange(data):
    a = list()
    for i in data:
        a += list(i)
    return a


class DataAugmentation:
    def __init__(self):
        pass

    def transform_landmark(self, img_data, trans_M):
        """
        :param img_data:  data
        :param trans_M: transformation matrix
        :return: transformed  data
        """
        image_data = np.concatenate([img_data, np.ones((img_data.shape[0], 1))], axis=1)
        transformed_data = np.dot(trans_M, image_data.T).T
        return transformed_data

    def transform_img(self, img, trans_M):
        """
        :param img:
        :param trans_M:
        :return:
        """
        transformed_img = cv2.warpAffine(img, trans_M, img.shape[0:2], borderValue=(255, 255, 255))
        return transformed_img

    def transformation(self, img, landmark, trans_M):
        transformed_img = self.transform_img(img, trans_M)
        transformed_landmark = self.transform_landmark(landmark, trans_M)
        return transformed_img, transformed_landmark

    def flip(self, img_data, landmark):
        ""
        """
        :param img: image data
        :param landmark: 
        :return: flipped image data
        """
        transformed_landmark = copy.deepcopy(landmark)
        transformed_landmark[:, 0] = img_data.shape[0] - transformed_landmark[:, 0]
        return np.fliplr(img_data), transformed_landmark

    def rotate(self, img_data, landmark, beta, direction='center'):
        """
        :param img_data: image data
        :param beta: rotate degree, beta >0: counterclockwise; beta <0: clockwise rotation
        :return: rotated image data
        """
        transform_M = np.array([[math.cos(beta), -math.sin(beta), 0],
                                [math.sin(beta), math.cos(beta), 0], [0, 0, 1]])
        transformed_landmark = self.transform_landmark(landmark, transform_M)
        center_M = self.move_accord_direction(landmark, transformed_landmark, direction=direction)
        Ntransform_M = np.dot(center_M, transform_M)

        return self.transformation(img_data, landmark, Ntransform_M[:2, :])

    def scale_sep(self, img_data, landmark, factor1, factor2, direction='upper_left'):
        """
        :param img_data:
        :param factor1:
        :param factor2:
        :return:
        """
        transform_M = np.array([[factor1, 0, 0], [0, factor2, 0]])
        transform_M = np.insert(transform_M, len(transform_M), values=[0, 0, 1], axis=0)
        transformed_landmark = self.transform_landmark(landmark, transform_M)
        center_M = self.move_accord_direction(landmark, transformed_landmark, direction=direction)
        Ntransform_M = np.dot(center_M, transform_M)
        return self.transformation(img_data, landmark, Ntransform_M[:2, :])

    def move(self, img_data, landmark, delta_x, delta_y):
        """
        :param img_data:
        :param delta_x:
        :param delta_y:
        :return:
        """
        transform_M = np.array([[1, 0, float(delta_x)], [0, 1, float(delta_y)]])
        return self.transformation(img_data, landmark, transform_M)

    def specfic_transform_landmark(self, landmark, trans_M, pointRange):
        transformed_landmark = copy.deepcopy(landmark)
        transformed_landmark[pointRange] = self.transform_landmark(transformed_landmark[pointRange], trans_M)
        return transformed_landmark

    def specific_scale(self, img_data, landmark, pointRange, factor1, factor2, direction='center'):
        """
        To control transformation range
        :param img_data:
        :param pointRange:
        :param factor1:
        :param factor2:
        :return:
        """
        transform_M = np.array([[factor1, 0, 0], [0, factor2, 0]])
        transformed_landmark = self.specfic_transform_landmark(landmark, transform_M, pointRange)
        center_M = self.move_accord_direction(landmark[pointRange], transformed_landmark[pointRange],
                                              direction=direction)
        transform_M = np.insert(transform_M, len(transform_M), values=[0, 0, 1], axis=0)
        Ntransform_M = np.dot(center_M, transform_M)
        Ntransformed_landmark = self.specfic_transform_landmark(landmark, Ntransform_M[:2, :], pointRange)
        transformed_img = self.thin_plate_spline(img_data, landmark, Ntransformed_landmark)
        return transformed_img, Ntransformed_landmark

    def move_accord_direction(self, landmark, transformed_landmark, direction='center'):
        if direction == 'upper_left':
            center_M = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
        elif direction == 'center':
            center_M = np.array([[1, 0, landmark[:, 0].mean() - transformed_landmark[:, 0].mean()],
                                 [0, 1, landmark[:, 1].mean() - transformed_landmark[:, 1].mean()], [0, 0, 1]])
        elif direction == 'upper':
            center_M = np.array([[1, 0, landmark[:, 0].mean() - transformed_landmark[:, 0].mean()],
                                 [0, 1, landmark[:, 1].min() - transformed_landmark[:, 1].min()], [0, 0, 1]])
        elif direction == 'under':
            center_M = np.array([[1, 0, landmark[:, 0].mean() - transformed_landmark[:, 0].mean()],
                                 [0, 1, landmark[:, 1].max() - transformed_landmark[:, 1].max()], [0, 0, 1]])
        return center_M

    # right
    def specific_move(self, img_data, landmark, pointRange, factor1, factor2):
        """
        :param img_data:
        :param pointRange:
        :param factor1:
        :param factor2:
        :return:
        """
        transform_M = np.array([[1, 0, factor1], [0, 1, factor2]])
        transformed_landmark = self.specfic_transform_landmark(landmark, transform_M, pointRange)

        transformed_img = self.thin_plate_spline(img_data, landmark, transformed_landmark)
        return transformed_img, transformed_landmark

    # right
    def thin_plate_spline(self, img, src, dst):
        """
        :param img: realistic image
        :param src: source landmark
        :param dst: target landmark
        :return: transformed realistic image (portrait)
        """
        coord = np.array([
            [0., 0.],
            [img.shape[0], 0.],
            [img.shape[0], img.shape[1]],
            [0., img.shape[1]]])
        coord_src = np.insert(src, 0, values=coord, axis=0).reshape(1, -1, 2)
        coord_dst = np.insert(dst, 0, values=coord, axis=0).reshape(1, -1, 2)
        N = 37
        matches = [cv2.DMatch(i, i, 0) for i in range(1, 2 * N + 1)]
        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.estimateTransformation(coord_dst, coord_src, matches)
        img = tps.warpImage(img)
        return img

    def complete_landmark(self, landmark):
        reverse_part = copy.deepcopy(landmark[1:16])
        reverse_part[:, 1] = 2 * min(landmark[0:17, 1]) - reverse_part[:, 1]
        NLandmark = np.insert(landmark, 17, values=reverse_part, axis=0)
        return NLandmark
