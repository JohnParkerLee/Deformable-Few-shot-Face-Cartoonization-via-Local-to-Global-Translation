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
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_type_dataset,make_json_dataset
from util import labelJson

def getfeats(featpath):
	trans_points = np.empty([5,2],dtype=np.int64)
	with open(featpath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		for ind,row in enumerate(reader):
			trans_points[ind,:] = list(map(int,map(eval,row)))
	return trans_points

def getSoft(size,xb,yb,boundwidth=5.0):
    xarray = np.tile(np.arange(0,size[1]),(size[0],1))
    yarray = np.tile(np.arange(0,size[0]),(size[1],1)).transpose()
    cxdists = []
    cydists = []
    for i in range(len(xb)):
        xba = np.tile(xb[i],(size[1],1)).transpose()
        yba = np.tile(yb[i],(size[0],1))
        cxdists.append(np.abs(xarray-xba))
        cydists.append(np.abs(yarray-yba))
    xdist = np.minimum.reduce(cxdists)
    ydist = np.minimum.reduce(cydists)
    manhdist = np.minimum.reduce([xdist,ydist])
    im = (manhdist+1) / (boundwidth+1) * 1.0
    im[im>=1.0] = 1.0
    return im



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
        self.A_fea_path = os.path.join(self.A_feat_dir, self.B_paths.split('/')[-1])
        self.B_fea_path = os.path.join(self.B_feat_dir, *self.B_paths.split('/')[-2::])
        A = self.loadImg(self.A_paths).convert('RGB')
        B = self.loadImg(self.B_paths).convert('RGB')




        # read a image given a random integer index
        # A = Image.open(self.A_paths).convert('RGB').resize((self.loadSize, self.loadSize), Image.BICUBIC)
        # B = Image.open(self.B_paths).convert('RGB').resize((self.loadSize, self.loadSize), Image.BICUBIC)
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        # B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.loadSize - self.fineSize - 1))
        h_offset = random.randint(0, max(0, self.loadSize - self.fineSize - 1))

        A = A[:, h_offset:h_offset + self.fineSize, w_offset:w_offset + self.fineSize]  # C,H,W
        B = B[:, h_offset:h_offset + self.fineSize, w_offset:w_offset + self.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.which_direction == 'BtoA':
            input_nc = self.output_nc
            output_nc = self.input_nc
        else:
            input_nc = self.input_nc
            output_nc = self.output_nc

        flipped = False
        if (not self.no_flip) and random.random() < 0.5:
            flipped = True
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        item = {'A': A, 'B': B,
                'A_paths': self.A_paths, 'B_paths': self.B_paths}

        regions = ['eyel', 'eyer', 'nose', 'mouth']
        basen = os.path.basename(self.A_paths)[:-4] + '.txt'
        featdir = self.lm_dir
        featpath = os.path.join(featdir, basen)
        feats = getfeats(featpath)
        # if flipped:
        #     for i in range(5):
        #         feats[i, 0] = self.fineSize - feats[i, 0] - 1
        #     tmp = [feats[0, 0], feats[0, 1]]
        #     feats[0, :] = [feats[1, 0], feats[1, 1]]
        #     feats[1, :] = tmp
        mouth_x = int((feats[3, 0] + feats[4, 0]) / 2.0)
        mouth_y = int((feats[3, 1] + feats[4, 1]) / 2.0)
        ratio = self.fineSize / 256
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
        if self.soft_border:
            soft_border_mask4 = []
            for i in range(4):
                xb = [np.zeros(rhs[i]), np.ones(rhs[i]) * (rws[i] - 1)]
                yb = [np.zeros(rws[i]), np.ones(rws[i]) * (rhs[i] - 1)]
                soft_border_mask = getSoft([rhs[i], rws[i]], xb, yb)
                soft_border_mask4.append(torch.Tensor(soft_border_mask).unsqueeze(0))
                item['soft_' + regions[i] + '_mask'] = soft_border_mask4[i]
        for i in range(4):
            item[regions[i] + '_A'] = A[:, int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
                                      int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)]
            item[regions[i] + '_B'] = B[:, int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
                                      int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)]
            if self.soft_border:
                item[regions[i] + '_A'] = item[regions[i] + '_A'] * soft_border_mask4[i].repeat(
                    int(input_nc / output_nc), 1, 1)
                item[regions[i] + '_B'] = item[regions[i] + '_B'] * soft_border_mask4[i]

        mask = torch.ones(B.shape)  # mask out eyes, nose, mouth
        for i in range(4):
            mask[:, int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
            int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)] = 0
        if self.soft_border:
            imgsize = self.fineSize
            maskn = mask[0].numpy()
            masks = [np.ones([imgsize, imgsize]), np.ones([imgsize, imgsize]), np.ones([imgsize, imgsize]),
                     np.ones([imgsize, imgsize])]
            masks[0][1:] = maskn[:-1]
            masks[1][:-1] = maskn[1:]
            masks[2][:, 1:] = maskn[:, :-1]
            masks[3][:, :-1] = maskn[:, 1:]
            masks2 = [maskn - e for e in masks]
            bound = np.minimum.reduce(masks2)
            bound = -bound
            xb = []
            yb = []
            for i in range(4):
                xbi = [int(center[i, 0] - rws[i] / 2), int(center[i, 0] + rws[i] / 2 - 1)]
                ybi = [int(center[i, 1] - rhs[i] / 2), int(center[i, 1] + rhs[i] / 2 - 1)]
                for j in range(2):
                    maskx = bound[:, xbi[j]]
                    masky = bound[ybi[j], :]
                    xb += [(1 - maskx) * 10000 + maskx * xbi[j]]
                    yb += [(1 - masky) * 10000 + masky * ybi[j]]
            soft = 1 - getSoft([imgsize, imgsize], xb, yb)
            soft = torch.Tensor(soft).unsqueeze(0)
            mask = (torch.ones(mask.shape) - mask) * soft + mask

        return item

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def loadImg(self, data):
        self.labelJson.load(data)
        return self.labelJson.imageData, np.array(self.labelJson.shapes[0]['points'])


