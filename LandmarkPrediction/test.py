import os
from itertools import islice

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import html
from util import util
from util.visualizer import save_images
import numpy as np
# options
opt = TestOptions().parse()
opt.num_threads = 1  # test code only supports num_threads=1
opt.batch_size = 1  # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle
opt.sync = False

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
save_dir = os.path.join(opt.results_dir, opt.part_class)
ori_save_dir = os.path.join(opt.results_dir, opt.part_class + '_ori')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(ori_save_dir):
    os.mkdir(ori_save_dir)

webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))


def inverse_norm(opt, landmark):
    minx = util.part[opt.part_class][0]
    maxx = util.part[opt.part_class][1]
    miny = util.part[opt.part_class][2]
    maxy = util.part[opt.part_class][3]
    # minx = 117
    # miny = 161
    # maxx = 394
    # maxy = 423
    landmark[:, 0] = landmark[:, 0] * (maxx - minx) + minx
    landmark[:, 1] = landmark[:, 1] * (maxy - miny) + miny
    return landmark

num = 0
# test stage
res_face = np.zeros((int(opt.input_nc/2),2))
average_face = np.zeros((int(opt.input_nc/2),2))
for i, data in enumerate(islice(dataset, opt.num_test)):
    num += 1
    model.k_shot_input(data)
    # print(data['data_B'][opt.part_class].shape)
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    real_A, real_B, fake_A2B = model.k_shot_test()
    real_A_res = inverse_norm(opt, real_A[0].reshape(int(opt.input_nc / 2), 2).cpu().detach().numpy())
    real_A = util.plot_landmark(real_A_res, opt.load_size)
    # real_A2B = util.plot_landmark(inverse_norm(opt, real_A2B[0].reshape(int(opt.input_nc/2), 2).cpu().detach().numpy()),
    #                                opt.load_size)
    np.savetxt(os.path.join(ori_save_dir,data['A_paths'][0].split('/')[-1]), real_A_res, fmt="%d")
    fake_A2B_res = inverse_norm(opt, fake_A2B[0].reshape(int(opt.input_nc / 2), 2).cpu().detach().numpy())
    res_face += fake_A2B_res
    np.savetxt(os.path.join(save_dir, data['A_paths'][0].split('/')[-1]), fake_A2B_res, fmt="%d")
    fake_A2B = util.plot_landmark(fake_A2B_res, opt.load_size)
    images = [real_A, fake_A2B]
    names = ['real_A', 'fake_A2B']
    for j in range(len(real_B[0])):
        if i == 0:
            average_face += inverse_norm(opt, real_B[0][j].reshape(int(opt.input_nc / 2), 2).cpu().detach().numpy())
        images.append(
            util.plot_landmark(inverse_norm(opt, real_B[0][j].reshape(int(opt.input_nc / 2), 2).cpu().detach().numpy()),
                               opt.load_size))
        names.append('style image%2.2d' % j)


    img_path = data['A_paths'][0].split('/')[-1][:-5]#'input_%3.3d' % i
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

average_face /= len(real_B[0])
res_face /= num

res = np.sum(np.sqrt(np.sum(np.square(res_face-average_face),1)))
print(res, res/(opt.input_nc/2.0))
webpage.save()
