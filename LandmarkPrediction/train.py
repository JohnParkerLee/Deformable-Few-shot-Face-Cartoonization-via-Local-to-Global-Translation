"""General-purpose training script for image-to-image translation.
This script works for various models (with option '--model': e.g., bicycle_gan, pix2pix, test) and
different datasets (with option '--dataset_mode': e.g., aligned, single).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
Example:
    Train a BiCycleGAN model:
        python train.py --dataroot ./datasets/facades --name facades_bicyclegan --model bicycle_gan --direction BtoA
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/train_options.py for more training options.
"""
import copy
import time

from tensorboardX import SummaryWriter

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util import util

writer = SummaryWriter('0526/5_points')


def inverse_norm(opt, landmark):
    # minx = 117
    # miny = 161
    # maxx = 394
    # maxy = 423
    minx = util.part[opt.part_class][0]
    maxx = util.part[opt.part_class][1]
    miny = util.part[opt.part_class][2]
    maxy = util.part[opt.part_class][3]
    landmark[:, 0] = landmark[:, 0] * (maxx - minx) + minx
    landmark[:, 1] = landmark[:, 1] * (maxy - miny) + miny
    return landmark


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    if opt.validate_freq > 0:
        validate_opt = copy.deepcopy(opt)
        validate_opt.phase = 'val'
        # validate_opt.serial_batches = True
        # validate_opt.batch_size = 1
        # validate_opt.k = 4
        val_data_loader = create_dataset(validate_opt)
        val_dataset_size = len(val_data_loader)  # get the number of images in the dataset.
        print('The number of training images = %d' % val_dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    for epoch in range(opt.epoch_count,
                       opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            # visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            if not model.is_train():  # if this batch of input data is enough for training.
                print('skip this batch')
                continue
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                for label, image in model.get_current_visuals().items():
                    if label == '':
                        for index in range(len(image[0])):
                            img = util.plot_landmark(inverse_norm(opt, image[0][index].reshape(int(opt.input_nc / 2),
                                                                                               2).cpu().detach().numpy()),
                                                     opt.load_size)
                            # image_numpy = util.tensor2im(image.unsqueeze(dim=0))
                            img = img.transpose([2, 0, 1])
                            img_names = label + '/' + str(index)
                            writer.add_image('img/' + img_names, img, total_iters)
                    else:
                        image = util.plot_landmark(
                            inverse_norm(opt, image[0].reshape(int(opt.input_nc / 2), 2).cpu().detach().numpy()),
                            opt.load_size)
                        # image_numpy = util.tensor2im(image.unsqueeze(dim=0))
                        image = image.transpose([2, 0, 1])
                        writer.add_image('img/' + label, image, total_iters)
                # model.compute_visuals()
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                losses = model.get_current_losses()
                for name in losses:
                    writer.add_scalar('%s' % name, losses[name], total_iters)
                # visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            if opt.validate_freq > 0:
                if total_iters % opt.validate_freq == 0:
                    model.eval()
                    for j, val_data in enumerate(val_data_loader):
                        model.set_input(val_data)
                        real_A, real_B, fake_A2B, fake_B2A = model.test()
                        real_A = util.plot_landmark(
                            inverse_norm(opt, real_A[0].reshape(int(opt.input_nc / 2), 2).cpu().detach().numpy()),
                            opt.load_size)
                        real_B = util.plot_landmark(
                            inverse_norm(opt, real_B[0].reshape(int(opt.input_nc / 2), 2).cpu().detach().numpy()),
                            opt.load_size)
                        # real_A2B = util.plot_landmark(inverse_norm(opt,real_A2B[0].reshape(int(opt.input_nc/2), 2).cpu().detach().numpy()),
                        #                               opt.load_size)
                        fake_A2B = util.plot_landmark(
                            inverse_norm(opt, fake_A2B[0].reshape(int(opt.input_nc / 2), 2).cpu().detach().numpy()),
                            opt.load_size)
                        fake_B2A = util.plot_landmark(
                            inverse_norm(opt, fake_B2A[0].reshape(int(opt.input_nc / 2), 2).cpu().detach().numpy()),
                            opt.load_size)
                        writer.add_image('val/val_real_A', real_A.transpose([2, 0, 1]), total_iters)
                        writer.add_image('val/val_real_B', real_B.transpose([2, 0, 1]), total_iters)
                        # writer.add_image('val_real_A2B', real_A2B.transpose([2, 0, 1]), total_iters)
                        writer.add_image('val/val_fake_A2B', fake_A2B.transpose([2, 0, 1]), total_iters)
                        writer.add_image('val/val_fake_B2A', fake_B2A.transpose([2, 0, 1]), total_iters)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.
    writer.close()
