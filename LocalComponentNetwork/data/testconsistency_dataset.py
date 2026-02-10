import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import random
import cv2
import torch
import numpy as np
import csv
import torchvision.transforms.functional as TF
class TestconsistencyDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain Ar '/path/to/data/trainA'
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
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform = get_transform(self.opt, grayscale=(output_nc == 1))
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.EYE_H = 72
        self.EYE_W = 64
        self.NOSE_H = 40
        self.NOSE_W = 48
        self.MOUTH_H = 40
        self.MOUTH_W = 64
        self.soft_border = False
        self.fineSize = 512.0
        self.no_flip = True
        # self.k = opt.k
        self.lm_dir = os.path.join(opt.dataroot, 'feat')
        self.mask_dir = os.path.join(opt.dataroot, 'mask')
        self.regions = ['eyel', 'eyer', 'nose', 'mouth']
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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')
        img_flur_B = img_B.filter(ImageFilter.GaussianBlur(radius=(random.random()*0.5+1)))
        item_A = self.getcomponent('A', img_A, A_path)
        item_B = self.getcomponent('B', img_B, B_path)


        return {'A': item_A[self.component+'_A'], 'B': item_B[self.component+'_B'], 'A_paths': A_path, 'B_paths': B_path}

    def getcomponent(self, class_name, img, path):
        item = {}
        img = self.transform(img)
        basen = os.path.join(*path.split('/')[-2::])[:-4] + '.txt'
        featdir = self.lm_dir
        featspath = os.path.join(featdir, basen)
        feats = self.getfeats(featspath)
        # is
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
            item[self.regions[i] + '_' + class_name] = img[:, int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
                                      int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)]
        mask = torch.ones(img.shape)  # mask out eyes, nose, mouth
        for i in range(4):
            mask[:, int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
            int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)] = 0
        mask = TF.resize(((img / 2 + 0.5) * mask )* 2 - 1, 256)
        img = TF.resize(img, 256)
        item['mask' + '_' + class_name] = mask
        item['img_'+class_name] = img
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
