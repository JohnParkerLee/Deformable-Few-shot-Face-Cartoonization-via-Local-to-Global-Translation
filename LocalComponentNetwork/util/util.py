"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = normPRED(image_tensor[0]).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = np.transpose(image_numpy, (1,2,0))*255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))
def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
MISSING_VALUE = -1
# def load_pose_cords_from_strings(y_str, x_str):
#     y_cords = json.loads(y_str)
#     x_cords = json.loads(x_str)
#     return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def cords_to_map(cords, img_size, sigma=6):
    result = torch.zeros(img_size + cords.shape[0:1], dtype=torch.float)
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
        xxyy = torch.cat([xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1)], dim = 1)
        result[..., i] = torch.exp(-((xxyy[:,1] - point[0]) ** 2 + (xxyy[:,0] - point[1]) ** 2) / (2 * sigma ** 2)).reshape(img_size)
        # result[..., i] = np.where(((yy - point[0]) ** 2 + (xx - point[1]) ** 2) < (sigma ** 2), 1, 0)
    return result
def norm(points_int, width, height):
    """
    将像素点坐标归一化至 -1 ~ 1
    """
    # points_int_clone = torch.from_numpy(points_int).detach().float().to(device)
    x = ((points_int * 2)[..., 0] / (width - 1) - 1)
    y = ((points_int * 2)[..., 1] / (height - 1) - 1)
    return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)

def thin_plate_spline_torch(img, src, dst):
    # img = img.squeeze(0)
    h, w = img.shape[2], img.shape[3]
    coord = torch.tensor([[
        [0., 0.],
        [h, 0.],
        [h, w],
        [0., w]]])
    coord_src = torch.cat([src,coord], 1)
    coord_dst = torch.cat([dst,coord], 1)

    ten_source = norm(coord_src, w, h)
    ten_target = norm(coord_dst, w, h)

    tps = TPS()
    warped_grid = tps(ten_target[None, ...], ten_source[None, ...], w, h, img.device) # 这个输入的位置需要归一化，所以用norm
    ten_wrp = torch.grid_sampler_2d(img, warped_grid, 0, 0, True)
    # new_img_torch = np.array((ten_wrp[0].cpu()))
    return ten_wrp

class TPS(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y, w, h, device):

        """ 计算grid"""
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        grid = grid.view(-1, h * w, 2)

        """ 计算W, A"""
        n, k = X.shape[:2]
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)

        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + eps)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        Q = torch.solve(Z, L)[0]
        W, A = Q[:, :k], Q[:, k:]

        """ 计算U """
        eps = 1e-9
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + eps)

        """ 计算P """
        n, k = grid.shape[:2]
        device = grid.device
        P = torch.ones(n, k, 3, device=device)
        P[:, :, 1:] = grid

        # grid = P @ A + U @ W
        grid = torch.matmul(P, A) + torch.matmul(U, W)
        return grid.view(-1, h, w, 2)