"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_type_dataset, make_txt_dataset
from util import labelJson
import os
import random
import numpy as np
import PIL
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from util import util
import time

landmark_items = {'head': [range(0, 17)], 'L_eyebrows': range(17, 22), 'R_eyebrows': range(22, 27),
                  'eyebrows': range(17, 27), 'nose': [range(27, 36)], 'L_eyes': [range(36, 42)],
                  'R_eyes':[range(42, 48)], 'eyes': range(36, 48), 'mouth': [range(48, 68)]}
class TestFinetuneDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     """Add new dataset-specific options, and rewrite default values for existing options.
    #
    #     Parameters:
    #         parser          -- original option parser
    #         is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
    #
    #     Returns:
    #         the modified parser.
    #     """
    #     parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
    #     parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
    #     return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.labelJson = labelJson()
        self.style_data_dir = os.path.join(opt.dataroot, opt.style_data_paths)
        self.content_data_dir = os.path.join(opt.dataroot, opt.content_data_paths)
        self.k = opt.k
        self.img_name = ""
        self.content_data = make_txt_dataset(self.content_data_dir)
        self.style_data = make_txt_dataset(self.style_data_dir)
        # get the image paths of your dataset;
        # self.landmark_types_paths = sorted(make_type_dataset(self.landmark_types))  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # self.content_data_paths = sorted(make_json_dataset(self.content_data_dir))
        # self.style_data_paths = sorted(make_type_dataset(self.style_data_dir))
        self.shape = opt.load_size
        self.part_class = opt.part_class
        self.len = len(self.content_data)
        self.part = util.part
        # self.mode = mode
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)
    def normalization(self, data_json, item):
        minx = 134
        miny = 174
        maxx = 386
        maxy = 408
        # data = data_json[:, util.concatRange(landmark_items[item]), :]
        data_json[0, :, 0] = (data_json[0, :, 0] - minx) / (maxx - minx)
        data_json[0, :, 1] = (data_json[0, :, 1] - miny) / (maxy - miny)
        data = data_json.reshape(1, -1)
        return data
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # start_time = time.time()
        # minx = 194
        # miny = 206
        # maxx = 338
        # maxy = 350
        # minx = util.part[self.part_class][0]
        # maxx = util.part[self.part_class][1]
        # miny = util.part[self.part_class][2]
        # maxy = util.part[self.part_class][3]
        minx = 134
        miny = 174
        maxx = 386
        maxy = 408
        data_B = []
        np.random.seed()
        np.random.shuffle(self.style_data)
        content_path = self.content_data[index%self.len]
        # style_path = np.random.choice(self.style_data)
        k_shot_data = np.random.choice(self.style_data, self.k, replace=False)
        data_A = ToTensor()(self.load_json_data(content_path))
        data_A= self.normalization(data_A, self.part_class)
        for items in k_shot_data:
            # data_B.append(self.plot_landmark(self.load_json_data(items), self.shape))
            data_B_temp = ToTensor()(self.load_json_data(items))
            data_B.append(self.normalization(data_B_temp, self.part_class))
        # data_B = ToTensor()(self.load_json_data(style_path))
        # data_B = self.normalization(data_B, self.part_class)
        # data_AB = ToTensor()(self.load_json_data(os.path.join(self.style_data_dir, content_path.split('/')[-1])))
        # data_AB = self.normalization(data_AB, self.part_class)
        # data_A_style = ToTensor()(self.load_json_data(np.random.choice(self.content_data)))
        # data_A_style = self.normalization(data_A_style, self.part_class)
        return {'data_A': data_A, 'data_B': torch.cat(data_B, dim=0).reshape(len(data_B), 1, -1)}

    def load_json_data(self, data):
        # data = np.loadtxt(data)
        # landmark = np.array(data)
        # points = []
        # points.append(np.mean(landmark[util.concatRange(landmark_items['L_eyes'])], axis=0))
        # points.append(np.mean(landmark[util.concatRange(landmark_items['R_eyes'])], axis=0))
        # points.append(landmark[30])
        # points.append(landmark[48])
        # points.append(landmark[54])
        return np.array(np.loadtxt(data))

    def __len__(self):
        """Return the total type of images."""
        return len(self.content_data)

    def plot_landmark(self, landmarks, shape):
        """
        Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
        matching the landmarks.

        The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
        plot to an image.

        Things to watch out for:
        * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
        only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
        * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
        * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

        :param frame: Image with a face matching the landmarks.
        :param landmarks: Landmarks of the provided frame,
        :return: RGB image with the landmarks as a Pillow Image.
        """
        dpi = 75
        fig = plt.figure(figsize=(shape / dpi, shape / dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.axis('off')
        plt.imshow(np.zeros((shape, shape)))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Head
        ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='#FFFFFF', lw=4)
        # Eyebrows
        ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='#FFFFFF', lw=4)
        ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='#FFFFFF', lw=4)
        # Nose
        ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='#FFFFFF', lw=4)
        ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='#FFFFFF', lw=4)
        # Eyes
        ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='#FFFFFF', lw=4)
        ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='#FFFFFF', lw=4)
        # Mouth
        ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='#FFFFFF', lw=4)
        # # Head
        # ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
        # # Eyebrows
        # ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
        # ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
        # # Nose
        # ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
        # ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
        # # Eyes
        # ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
        # ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
        # # Mouth
        # ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)
        fig.canvas.draw()
        # import io
        # buffer = io.BytesIO()  # using buffer,great way!
        # # 把plt的内容保存在内存中
        # plt.savefig(buffer, format='png')
        data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB',
                                    0, 1)
        plt.close(fig)
        # return np.expand_dims(np.asarray(data)[:,:,1], 2)
        # return ToTensor()(np.asarray(data)[:,:,1])
        return np.asarray(data)

    def set_img(self, img_name):
        self.img_name = img_name
