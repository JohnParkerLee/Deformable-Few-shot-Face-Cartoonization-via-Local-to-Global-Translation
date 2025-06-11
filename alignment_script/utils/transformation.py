import os
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from os.path import join, getsize
import math
class Img:
    def __init__(self, rows, cols, center=[0, 0]):
        # self.src = image  # 原始像
        self.rows = rows  # 原始图像的行
        self.cols = cols  # 原始图像的列
        self.center = center  # 旋转中心，默认是[0,0]

    def Move(self, delta_x, delta_y):  # 平移
        # delta_x>0左移，delta_x<0右移
        # delta_y>0上移，delta_y<0下移
        self.transform = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])

    def Zoom(self, factor):  # 缩放
        # factor>1表示缩小；factor<1表示放大
        self.transform = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])

    def ZoomSep(self, factor1, factor2):
        self.transform = np.array([[factor1, 0, 0], [0, factor2, 0], [0, 0, 1]])

    def Horizontal(self):  # 水平镜像
        self.transform = np.array([[1, 0, 0], [0, -1, self.cols - 1], [0, 0, 1]])

    def Vertically(self):  # 垂直镜像
        self.transform = np.array([[-1, 0, self.rows - 1], [0, 1, 0], [0, 0, 1]])

    def Rotate(self, beta):  # 旋转
        # beta>0表示逆时针旋转；beta<0表示顺时针旋转
        self.transform = np.array([[math.cos(beta), -math.sin(beta), 0],
                                   [math.sin(beta), math.cos(beta), 0],
                                   [0, 0, 1]])