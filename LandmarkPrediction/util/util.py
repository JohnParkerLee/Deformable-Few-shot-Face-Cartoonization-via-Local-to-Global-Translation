from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import pickle
import json
# landmark_items = {'head': [range(0, 17)], 'L_eyebrows': range(17, 22), 'R_eyebrows': range(22, 27),
#                   'eyebrows': range(17, 27), 'nose': [range(27, 36)], 'L_eyes': [range(17, 22), range(36, 42)],
#                   'R_eyes': [range(22, 27), range(42, 48)], 'eyes': range(36, 48), 'mouth': [range(48, 68)]}
landmark_items = {'head': [range(0, 17)], 'L_eyebrows': range(17, 22), 'R_eyebrows': range(22, 27),
                  'eyebrows': range(17, 27), 'nose': [range(27, 36)], 'L_eyes': [range(36, 42)],
                  'R_eyes': [range(42, 48)], 'eyes': range(36, 48), 'mouth': [range(48, 68)]}


# part_item = ['head','L_eyes','R_eyes','nose','mouth']
part_item = ['head', 'points_5']
input_nc = {'head': 34, 'L_eyes': 22, 'R_eyes': 22, 'nose': 18, 'mouth': 40, 'points_5':10}
part = {'L_eyes': (134, 262, 174, 318),
             'R_eyes': (258, 386, 174, 318),
             'nose': (194, 338, 206, 350),
             'mouth': (170, 346, 328, 408),
             'head': (0, 512, 0, 512),
            'points_5':(100,370,140,450)}
def tensor2im(input_image, imtype=np.uint8):
    """"Convert a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def concatRange(data):
    a = list()
    for i in data:
        a += list(i)
    return a

def tensor2vec(vector_tensor):
    numpy_vec = vector_tensor.data.cpu().numpy()
    if numpy_vec.ndim == 4:
        return numpy_vec[:, :, 0, 0]
    else:
        return numpy_vec


def pickle_load(file_name):
    data = None
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def pickle_save(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


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


def interp_z(z0, z1, num_frames, interp_mode='linear'):
    zs = []
    if interp_mode == 'linear':
        for n in range(num_frames):
            ratio = n / float(num_frames - 1)
            z_t = (1 - ratio) * z0 + ratio * z1
            zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    if interp_mode == 'slerp':
        z0_n = z0 / (np.linalg.norm(z0) + 1e-10)
        z1_n = z1 / (np.linalg.norm(z1) + 1e-10)
        omega = np.arccos(np.dot(z0_n, z1_n))
        sin_omega = np.sin(omega)
        if sin_omega < 1e-10 and sin_omega > -1e-10:
            zs = interp_z(z0, z1, num_frames, interp_mode='linear')
        else:
            for n in range(num_frames):
                ratio = n / float(num_frames - 1)
                z_t = np.sin((1 - ratio) * omega) / sin_omega * z0 + np.sin(ratio * omega) / sin_omega * z1
                zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    return zs


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

def plot_landmark(landmarks, shape):
    import matplotlib.pyplot as plt
    import PIL
    from torchvision.transforms import ToTensor
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
    return np.asarray(data)
MISSING_VALUE = -1
# def load_pose_cords_from_strings(y_str, x_str):
#     y_cords = json.loads(y_str)
#     x_cords = json.loads(x_str)
#     return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

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