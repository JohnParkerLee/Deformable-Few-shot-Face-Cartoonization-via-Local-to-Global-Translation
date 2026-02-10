import numpy as np
import pandas as pd 
import json
import os 
#from utils import labelJson
import pickle

MISSING_VALUE = -1
# fix PATH
img_dir  = 'Desktop/PATN_data/B/test/50/'
# annotations_file = 'fashion_data/fasion-resize-annotation-train.csv' #pose annotation path
save_path = 'Desktop/PATN_data/B/test/50/' #path to store pose maps

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords. shape[0:1], dtype=np.float)
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        xxyy = np.c_[xx.ravel(), yy.ravel()]
        result[..., i] = np.exp(-((xxyy[:,1] - point[0]) ** 2 + (xxyy[:,0] - point[1]) ** 2) / (2 * sigma ** 2)).reshape(img_size)
        # result[..., i] = np.where(((yy - point[0]) ** 2 + (xx - point[1]) ** 2) < (sigma ** 2), 1, 0)
    return result

def compute_pose(image_dir, savePath):
    image_size = (512, 512)
    for root, dirs, _ in os.walk(image_dir):
        for directory in dirs:
            for file in os.listdir(os.path.join(root, directory)):
                #labJson = labelJson()
                #labJson.load(os.path.join(root, directory, file))
                #landmark = np.array(labJson.shapes[0]['points'])
                save_dir = os.path.join(savePath, directory)
                file_name = os.path.join(save_dir, file.split('.')[0]+'.npy')
                #heatmap = cords_to_map(landmark[:,::-1], image_size)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                # file = open(file_name,'wb')
                # pickle.dump(heatmap, file)
                # file.close()
                #np.save(file_name, heatmap)
if __name__ == '__main__':
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    compute_pose(img_dir, save_path)



