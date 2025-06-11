import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import PIL.Image

from utils import Preprocess
from utils import labelJson
from utils import Img
from utils import DataAugmentation
from utils import data_augmentation as data_augmentation_torch
from utils.detect_5point import detect_5points,detect_5points_MTCNN


def convertCv2PIL(image):
    return PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def convertPIL2Cv(image):
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

pre = Preprocess()

def get_landmark(args):
    labJson = labelJson()
    for idx, img_name in enumerate(tqdm(os.listdir(args.data_path))):
        # img = cv2.cvtColor(cv2.imread(os.path.join(args.data_path, img_name)), cv2.COLOR_BGR2RGB)
        img = PIL.Image.open(os.path.join(args.data_path, img_name))
        landmark = np.squeeze(pre.face_landmark_info(convertPIL2Cv(img))).tolist()
        # print(landmark.shape)
        if landmark is not None:
            # np.savetxt(os.path.join(args.save_path, img_name.split('.')[0]+'.txt'), landmark, fmt="%d")
            labJson.save_landmark_json(img, os.path.join(args.data_path, img_name), landmark, img_name, args.save_path)
        else:
            print(img_name)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def alignment(args):
    labJson = labelJson()
    # Set your dataset folder
    APD_dir = "/home/xxx/Desktop/APD"
    for img_name in enumerate(tqdm(os.listdir(args.data_path))):
        APD_img = cv2.imread(os.path.join(APD_dir, img_name[1].split('.')[0]+'.png'),cv2.IMREAD_GRAYSCALE)
        labJson.load(os.path.join(args.data_path, img_name[1]))
        imageAli, landAli = pre.process(convertPIL2Cv(labJson.imageData), np.array(labJson.shapes[0]['points'])) # alignment
        apdAli, _ = pre.process(APD_img, np.array(labJson.shapes[0]['points']))

        transfor = Img(imageAli.shape[0], imageAli.shape[1],[0,0])
        transfor.ZoomSep(512.0/transfor.rows, 512.0/transfor.cols)

        imageAli = convertCv2PIL(imageAli)

        apdAli = convertCv2PIL(apdAli)

        landAli = np.concatenate([landAli, np.ones((landAli.shape[0], 1))], axis=1)
        landmarks_resize = np.dot(transfor.transform, landAli.T).T
        landmarks_resize = landmarks_resize[:,:2]
        labJson.save_landmark_json(imageAli, labJson.imagePath, landmarks_resize.tolist(), img_name[1], args.save_path)
        labJson.save_landmark_json(apdAli, os.path.join(APD_dir, img_name[1].split('.')[0]+'.png'), landmarks_resize.tolist(), img_name[1], os.path.join(APD_dir,"json"))


def APD_to_json(args):
    labJson = labelJson()
    # Set your dataset folder
    APD_dir = "/home/xxx/Desktop/test/apd"
    for img_name in enumerate(tqdm(os.listdir(args.data_path))):
        APD_img = cv2.imread(os.path.join(APD_dir, img_name[1].split('.')[0]+'.png'),cv2.IMREAD_GRAYSCALE)
        labJson.load(os.path.join(args.data_path, img_name[1]))
        labJson.save_landmark_json(labJson.imageData, labJson.imagePath, labJson.shapes[0]['points'], img_name[1], args.save_path)
        labJson.save_landmark_json(convertCv2PIL(APD_img), os.path.join(APD_dir, img_name[1].split('.')[0]+'.png'), labJson.shapes[0]['points'], img_name[1], os.path.join(APD_dir,"json"))

# Data augmentation
def data_aug(args):
    import random
    import pandas as pd
    data_aug = DataAugmentation()

    labJson = labelJson()
    df = pd.DataFrame(columns=["rotate", "scale_x", "scale_y", "move_x", "move_y", "specific_scale_x_head",
                               "specific_scale_y_head", 'specific_scale_x_nose', 'specific_scale_y_nose',
                               'specific_scale_x_eye', 'specific_scale_y_eye', 'specific_scale_x_mouth',
                               'specific_scale_y_mouth', 'specific_move_y_head', 'specific_move_y_nose',
                               'specific_move_y_eye'])
    for i in range(1000):
        rotate = random.choice([-np.pi / 6.0, 0.0, np.pi / 6.0])
        scale_x = random.choice([0.8, 1.0, 1.2])
        scale_y = random.choice([0.8, 1.0, 1.2])
        move_x = random.choice([-20.0, 0.0, 20.0])
        move_y = random.choice([-20.0, 0.0, 20.0])
        specific_scale_x_head = random.choice([0.7, 1.0, 1.4])
        specific_scale_y_head = random.choice([0.7, 1.0, 1.2])
        specific_scale_x_nose = random.choice([0.6, 1.0, 1.4])
        specific_scale_y_nose = random.choice([0.8, 1.0, 1.2])
        specific_scale_x_eye = random.choice([0.5, 1.0, 1.5])
        specific_scale_y_eye = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])
        specific_scale_x_mouth = random.choice([0.6, 1.0, 1.4])
        specific_scale_y_mouth = random.choice([0.6, 1.0, 1.4])
        specific_move_y_head = random.choice([-20.0, 0.0, 20.0])
        specific_move_y_nose = random.choice([-10.0, 0.0, 10.0])
        specific_move_y_eye = random.choice([-25.0, 0.0, 25.0])
        for index, img_name in enumerate(tqdm(os.listdir(args.data_path))):
            print(index, img_name)
            labJson.load(os.path.join(args.data_path,img_name))
            img = convertPIL2Cv(labJson.imageData)
            landmark = np.array(labJson.shapes[0]['points'])
            landmark = data_aug.complete_landmark(landmark)
            _, landmark = data_aug.specific_scale(img, landmark, data_augmentation_torch.landmark_items_complete['head'], 1.0, specific_scale_y_head, direction='upper')
            _, landmark = data_aug.specific_scale(img, landmark, data_augmentation_torch.landmark_items_complete['head_All'], specific_scale_x_head, 1.0, direction='center')
            _, landmark = data_aug.specific_scale(img, landmark, data_augmentation_torch.landmark_items_complete['nose'], specific_scale_x_nose, specific_scale_y_nose, direction='center')
            _, landmark = data_aug.specific_scale(img, landmark, data_augmentation_torch.landmark_items_complete['eyes'], specific_scale_x_eye, specific_scale_y_eye, direction='upper')
            _, landmark = data_aug.specific_scale(img, landmark, data_augmentation_torch.landmark_items_complete['mouth'], specific_scale_x_mouth, specific_scale_y_mouth, direction='upper')
            _, landmark = data_aug.specific_move(img, landmark, data_augmentation_torch.landmark_items_complete['head'], 0.0, specific_move_y_head)
            _, landmark = data_aug.specific_move(img, landmark, data_augmentation_torch.landmark_items_complete['nose'], 0.0, specific_move_y_nose)
            _, landmark = data_aug.specific_move(img, landmark, data_augmentation_torch.concatRange([data_augmentation_torch.landmark_items_complete['eyes'], data_augmentation_torch.landmark_items_complete['eyebrows']]), 0.0, specific_move_y_eye)
            os.makedirs(args.save_path+'/'+'json/'+str(i), exist_ok=True)
            labJson.save_landmark_json(convertCv2PIL(img), labJson.imagePath, landmark[data_augmentation_torch.concatRange([range(0,17), range(32,83)])].tolist(), img_name,
                                       args.save_path+'/'+'json/'+str(i))
        df.loc[i] = [rotate, scale_x, scale_y, move_x, move_y, specific_scale_x_head,
                            specific_scale_y_head, specific_scale_x_nose, specific_scale_y_nose,
                            specific_scale_x_eye, specific_scale_y_eye, specific_scale_x_mouth,
                            specific_scale_y_mouth, specific_move_y_head, specific_move_y_nose,
                            specific_move_y_eye]
        # save the augmentation information
        if i%100 == 0:
            df.to_csv(args.save_path+'/result_aug.csv', sep=' ', header=True, index=True)
    df.to_csv(args.save_path + '/result_aug.csv', sep=' ', header=True, index=True)

def segmentation(args):
    for img_name in enumerate(tqdm(os.listdir(args.data_path))):
        img = cv2.imread(os.path.join(args.data_path, img_name[1]))
        segImage = pre.segmentation(img)
        cv2.imwrite(os.path.join(args.save_path, img_name[1]), segImage)

def fromJsonGetImg(args):
    labJson = labelJson()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    for img_name in enumerate(tqdm(os.listdir(args.data_path))):
        labJson.load(os.path.join(args.data_path, img_name[1]))
        labJson.imageData.save(os.path.join(args.save_path, img_name[1].split('.')[0]+'.png'))

def transToJson(args):
    from PIL import Image
    # path to landmark
    landmark_path = ''
    if landmark_path == '':
        raise ValueError('Landmark path is needed!')

    landmark_list = os.listdir(landmark_path)
    labJson = labelJson()
    img_dir = args.data_path
    os.makedirs(args.save_path + '/' + 'json', exist_ok=True)
    for item in landmark_list:
        img_path = os.path.join(img_dir, item.split('.')[0]+'.png')
        landmark = np.loadtxt(os.path.join(landmark_path, item))
        img = Image.open(img_path)
        labJson.save_landmark_json(img, img_path,
                                   landmark.tolist(),
                                   item,
                                   args.save_path + '/' + 'json')

def metric(args):
    # data_aug = DataAugmentation()
    minx = 1e10
    maxx = -1
    miny = 1e10
    maxy = -1
    type = 'mouth'
    labJson = labelJson()
    if type in ['L_eye', 'R_eye']:
        labelRange = data_augmentation_torch.concatRange([data_augmentation_torch.landmark_items['L_eyebrows'],data_augmentation_torch.landmark_items['L_eyes']])
    else:
        labelRange = data_augmentation_torch.landmark_items[type]

    for root, _, fnames in sorted(os.walk(args.data_path)):
        for fname in fnames:
            labJson.load(os.path.join(root, fname))
            landmark = np.array(labJson.shapes[0]['points'])
            if minx > landmark[labelRange][:,0].min():
                minx = landmark[labelRange][:,0].min()
            if miny > landmark[labelRange][:, 1].min():
                miny = landmark[labelRange][:, 1].min()
            if maxx<landmark[labelRange][:,0].max():
                maxx = landmark[labelRange][:,0].max()
            if maxy < landmark[labelRange][:, 1].max():
                maxy = landmark[labelRange][:, 1].max()
    print('minx',minx,'\nminy',miny,'\nmaxx',maxx,'\nmaxy',maxy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, help = 'landmark and alignment')
    parser.add_argument('--data_path', type=str, help='photo folder path')
    parser.add_argument('--save_path', type=str, help='save folder path')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    if (args.mode == 'landmark'):
        get_landmark(args)
    elif(args.mode == 'alignment'):
        alignment(args)
    elif (args.mode == 'apd2json'):
        APD_to_json(args)
    elif(args.mode == 'augmentation'):
        data_aug(args)
    elif(args.mode == 'segmentation'):
        segmentation(args)
    elif(args.mode=='metric'):
        metric(args)
    elif(args.mode == 'json_to_mat'):
        detect_5points(args.data_path, args.save_path)
    elif (args.mode == 'json_to_txt'):
        detect_5points(args.data_path, args.save_path, mode='txt')
    elif (args.mode == 'detect_mtcnn'):
        detect_5points_MTCNN(args.data_path, args.save_path)
    elif (args.mode == 'get_img'):
        fromJsonGetImg(args)
    elif (args.mode =='trans_to_json'):
        transToJson(args)
    else:
        raise ValueError("Wrong mode!")

