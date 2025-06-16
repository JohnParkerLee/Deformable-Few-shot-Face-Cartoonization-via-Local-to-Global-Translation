import os.path
import random
import torchvision.transforms as transforms
import torch
# from data.base_dataset import BaseDataset
# from data.image_folder import make_dataset
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
import cv2
import csv
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_type_dataset, make_json_dataset
from util import labelJson


class ComponentDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.labelJson = labelJson()
        self.A_dir = os.path.join(opt.dataroot, opt.A_paths)
        self.A_feat_dir = os.path.join(opt.dataroot, opt.A_feat_paths)
        self.B_dir = os.path.join(opt.dataroot, opt.B_paths)
        self.B_feat_dir = os.path.join(opt.dataroot, opt.B_feat_paths)
        self.B_data_paths = sorted(make_json_dataset(self.B_dir))
        self.B_data_paths = np.random.shuffle(self.B_data_paths)
        self.shape = opt.load_size
        self.load_size = opt.load_size
        self.fine_size = opt.load_size
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

        # self.transform = get_transform(,)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        self.B_paths = self.B_data_paths[index]
        self.A_paths = os.path.join(self.A_dir, self.B_paths.split('/')[-1])
        # self.A_fea_path = os.path.join(self.A_feat_dir, self.B_paths.split('/')[-1])
        # self.B_fea_path = os.path.join(self.B_feat_dir, *self.B_paths.split('/')[-2::])
        A = self.loadImage(self.A_paths).convert('RGB')
        B = self.loadImage(self.B_paths).convert('RGB')
        A_heatmap = np.load(os.path.join(self.A_feat_dir, self.B_paths.split('/')[-1]))
        B_heatmap = np.load(os.path.join(self.B_feat_dir, *self.B_paths.split('/')[-2::]))
        A_heatmap = torch.from_numpy(A_heatmap).float()  # h, w, c
        A_heatmap = A_heatmap.transpose(2, 0)  # c,w,h
        A_heatmap = A_heatmap.transpose(2, 1)  # c,h,w

        B_heatmap = torch.from_numpy(B_heatmap).float()
        B_heatmap = B_heatmap.transpose(2, 0)  # c,w,h
        B_heatmap = B_heatmap.transpose(2, 1)  # c,h,w

        transforms_params = get_params(self.opt, A.size)
        transform_params = get_transform(self.opt, )
        A_transform = self.transform(A, transforms_params, grayscale=(self.input_nc == 1))
        B_transform = self.transform(B, transforms_params, grayscale=(self.input_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        return {'A': A, 'A_heatmap': A_heatmap, 'B': B, 'B_heatmap': B_heatmap}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def loadImage(self, data):
        self.labelJson.load(data)
        return self.labelJson.imageData
