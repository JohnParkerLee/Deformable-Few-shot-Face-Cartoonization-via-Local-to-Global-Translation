import csv
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor



import copy
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from models import orilandmark_gan_model as landmark_model
# from . import networks
from models import orilandmark_gan_model
from util import util
import  cv2

class UnpairConsistencyDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A_new')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B' + opt.dataset)  # create a path '/path/to/data/trainB'
        self.dir_A_landmark = os.path.join(opt.dataroot, 'landmark_A')
        self.dir_B_landmark = os.path.join(opt.dataroot, 'landmark_B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image

        self.transform_res = [transforms.ToTensor()]
        self.transform_res += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform_res = transforms.Compose(self.transform_res)
        self.EYE_H = 80
        self.EYE_W = 64
        self.NOSE_H = 40
        self.NOSE_W = 48
        self.MOUTH_H = 40
        self.MOUTH_W = 64
        self.fineSize = 512.0
        self.no_flip = True

        self.lm_dir = os.path.join(opt.dataroot, 'feat')
        self.mask_dir = os.path.join(opt.dataroot, 'mask')
        self.regions = ['eyel', 'eyer', 'nose', 'mouth']
        # Can use alignment script to determinate the boundary. Please refer to readme for detail.
        self.normal_landmark = {'minx': 100, 'miny': 140, 'maxx': 370, 'maxy': 450}
        self.k = 5
        self.trans_crop = transforms.RandomCrop(size=self.opt.crop_size)
        self.component = opt.component

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # unpair_data
        np.random.seed()
        self.opt.preprocess = 'rotate'
        self.transform_params = get_params(self.opt, (512, 512))
        self.transform_B = get_transform(self.opt, params=self.transform_params, grayscale=(self.output_nc == 1))
        self.opt.preprocess = 'rotate_and_color'
        # self.transform_params = get_params(self.opt, (512, 512))
        self.transform_A = get_transform(self.opt, params=self.transform_params, grayscale=(self.input_nc == 1))

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = np.random.choice(range(self.B_size), size=1, replace=False)
        B_path = self.B_paths[index_B[0]]

        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')

        img_flur_B = img_B.filter(ImageFilter.GaussianBlur(radius=(random.random() * 0.5 + 3.0)))
        if self.component == 'mask' or self.component == 'img':
            self.crop_params = transforms.RandomCrop.get_params(torch.randn(3, 256, 256),
                                                                (self.opt.crop_size, self.opt.crop_size))
        self.unpair = True
        item_A = self.getcomponent('A', img_A, A_path)
        item_B = self.getcomponent('B', img_B, B_path)

        self.unpair = False
        item_flur_B = self.getcomponent('B', img_flur_B, B_path)

        return {'A': item_A[self.component + '_A'], 'B': item_B[self.component + '_B'],
                'flur_B': item_flur_B[self.component + '_B'], 'A_paths': A_path, 'B_paths': B_path}


    def getcomponent(self, class_name, img, path):
        resize_size = 256
        item = {}
        if class_name == 'A':
            img = self.transform_A(img)
        else:
            img = self.transform_B(img)
        basen = os.path.join(*path.split('/')[-2::])[:-4] + '.txt'
        featdir = self.lm_dir
        featspath = os.path.join(featdir, basen)
        feats = self.getfeats(featspath)

        mouth_x = int((feats[3, 0] + feats[4, 0]) / 2.0)
        mouth_y = int((feats[3, 1] + feats[4, 1]) / 2.0)
        # ratio = self.fineSize / 256
        ratio = 2
        EYE_H = int(self.EYE_H * ratio)
        EYE_W = int(self.EYE_W * ratio)
        NOSE_H = int(self.NOSE_H * ratio)
        NOSE_W = int(self.NOSE_W * ratio)
        MOUTH_H = int(self.MOUTH_H * ratio)
        MOUTH_W = int(self.MOUTH_W * ratio)
        center = torch.IntTensor([[feats[0, 0], feats[0, 1] - 4 * ratio], [feats[1, 0], feats[1, 1] - 4 * ratio],
                                  [feats[2, 0], feats[2, 1] - NOSE_H / 2 + 16 * ratio], [mouth_x, mouth_y]])
        item['center'] = center
        rhs = [EYE_H, EYE_H, NOSE_H, MOUTH_H]
        rws = [EYE_W, EYE_W, NOSE_W, MOUTH_W]
        for i in range(4):
            item[self.regions[i] + '_' + class_name] = img[:,
                                                       int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
                                                       int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)]
        mask = torch.ones(img.shape)  # mask out eyes, nose, mouth
        for i in range(4):
            mask[:, int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
            int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)] = 0
        return item

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def getfeats(self, featpath):
        trans_points = np.empty([5, 2], dtype=np.int64)
        with open(featpath, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for ind, row in enumerate(reader):
                trans_points[ind, :] = list(map(int, map(eval, row)))
        return trans_points