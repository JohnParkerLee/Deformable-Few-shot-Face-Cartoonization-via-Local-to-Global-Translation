# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:15:37 2020

@author: Shengshu
"""

from mtcnn import MTCNN
import cv2
import os
import csv
from .trans_to_labeljson import labelJson
from utils import data_augmentation_torch as DA
import shutil
import numpy as np
import scipy.io

def write_csv_file(path, head, data):
    try:
        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, dialect='excel', delimiter=' ')

            if head is not None:
                writer.writerow(head)

            for row in data:
                writer.writerow(row)

            print("Write a CSV file to path %s Successful." % path)
    except Exception as e:
        print("Write an CSV file to path: %s, Case: %s" % (path, e))


def detect_5points(load_path, save_path, mode = 'mat'):

    landmark5points = ['L_eyes', 'R_eyes', 'nose', 'mouth_left', 'mouth_right']
    for root, dirs, _ in os.walk(load_path):
        for directory in dirs:
            for file in os.listdir(os.path.join(root, directory)):
                labJson = labelJson()
                labJson.load(os.path.join(root, directory, file))
                landmark = np.array(labJson.shapes[0]['points'], dtype=int)
                save_dir = os.path.join(save_path, directory)
                if mode == 'mat':
                    save_name = os.path.join(save_dir, file.split('.')[0] + ".mat")
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    scipy.io.savemat(save_name, {'points': np.array(landmark, dtype=int)})
                else:
                    save_name = os.path.join(save_dir, file.split('.')[0]+".txt")
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    write_csv_file(save_name, None, np.array(landmark, dtype=int))
                #write_csv_file(save_name, None, landmark)
                # scipy.io.savemat(save_name, {'points': np.array(landmark)})
                # break
                # points.append((np.mean(landmark[DA.landmark_items['L_eyes']], axis = 0)+np.mean(landmark[range(17,22)],axis=0))/2.0)
                # points.append((np.mean(landmark[DA.landmark_items['R_eyes']], axis=0) + np.mean(landmark[range(22, 27)],
                #                                                                                 axis=0)) / 2.0)
                # points.append(((landmark[17] + landmark[36]) / 2 + (landmark[21] + landmark[39]) / 2.0) / 2.0)
                # points.append(((landmark[22] + landmark[42]) / 2 + (landmark[26] + landmark[45]) / 2.0) / 2.0)
                # points.append(np.mean(landmark[DA.landmark_items['L_eyes']], axis=0))
                # points.append(np.mean(landmark[DA.landmark_items['R_eyes']], axis=0))
                # points.append((landmark[37]+landmark[38])/2.0)
                # points.append((landmark[43] + landmark[44]) / 2.0)
                # points.append(landmark[30])
                # points.append(landmark[48])
                # points.append(landmark[54])
                # for items in landmark5points:
                #     if items in ['mouth_left', 'mouth_right']:
                #         points.append(landmark[48])
                #         points.append(landmark[54])
                #         break
                #     elif items=='nose':
                #         print('nose')
                #         points.append(landmark[30])
                #     elif items in ['L_eyes', 'R_eyes']:
                #         points.append(((landmark[17]+landmark[36])/2+(landmark[21]+landmark[39])/2.0)/2.0)
                #         points.append(((landmark[22] + landmark[42]) / 2 + (landmark[26] + landmark[45]) / 2.0) / 2.0)
                #     else:
                #         points.append(np.mean(landmark[DA.landmark_items[items]], axis = 0))
                # for items in landmark5points:
                #     points.append(face_5points[items])

                # save_dir = os.path.join(save_path, directory)
                # save_name = os.path.join(save_dir, file.split('.')[0]+".txt")
                # if not os.path.exists(save_dir):
                #     os.mkdir(save_dir)
                # write_csv_file(save_name, None, points)


def detect_5points_MTCNN(load_path, save_path):
    detector = MTCNN()

    landmark5points = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
    for root, dirs, _ in os.walk(load_path):
        for directory in dirs:
            for file in os.listdir(os.path.join(root, directory)):
                imgs = cv2.imread(os.path.join(root, directory, file))
                face_5points = detector.detect_faces(imgs)
                if len(face_5points) == 0:
                    shutil.move(os.path.join(root, directory, file), os.path.join(save_path, file))
                    print(file + "fail")
                    continue
                points = []
                for item in landmark5points:
                    points.append(np.array(face_5points[0]['keypoints'][item]))
                save_dir = os.path.join(save_path, directory)
                save_name = os.path.join(save_dir, file.split('.')[0]+".mat")
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                scipy.io.savemat(save_name, {'points':np.array(points)})
                # write_csv_file(save_name, None, points)