import itertools

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from util.image_pool import ImagePool
from . import orilandmark_gan_model as landmark_model
from . import orilandmark_gan_model
from . import networks
from . import sw_loss
from .base_model import BaseModel
import torchvision.transforms.functional as TF
import os
import numpy as np
import csv
from U_2_Net.model import U2NET
# torch.autograd.set_detect_anomaly(True)

'''
Calculate warped image using control point manipulation on a thin plate (TPS)
Calculate warped image using control point manipulation on a thin plate (TPS)
Based on Herve Lombaert's 2006 web article
"Manual Registration with Thin Plates" 
(https://profs.etsmtl.ca/hlombaert/thinplates/)
Implementation by Yucheol Jung <ycjung@postech.ac.kr>
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CombineModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.【-
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt.k = 5
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['G_Eyer', 'G_Nose', 'G_Mouth']
        self.visual_names = ['img_A', 'result', 'fake_Eyel', 'fake_Eyer', 'fake_Nose', 'fake_Mouth', 'fake_Hair',
                              'real_A_Eyel', 'real_A_Eyer', 'real_A_Nose', 'real_A_Mouth', 'real_A_Hair']
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_Eyel = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_Eyer = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_Nose = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_Mouth = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        import copy
        landmark_opt = copy.deepcopy(opt)
        landmark_opt.isTrain = False
        self.landmark_predict_5 = landmark_model.OriLandmarkGanModel(landmark_opt, pretrain_G_content=opt.pretrain_G_content_5,
                                                             pretrain_G_style=opt.pretrain_G_style_5, input_nc = 10, output_nc = 10)

        self.landmark_predict_17 = orilandmark_gan_model.OriLandmarkGanModel(landmark_opt, pretrain_G_content=opt.pretrain_G_content_17,
                                                             pretrain_G_style=opt.pretrain_G_content_17)
        self.landmark_predict_5.eval()
        self.landmark_predict_17.eval()
        self.u2net = U2NET(3,1)
        self.u2net.load_state_dict(torch.load('U_2_Net/saved_models/u2net_portrait/u2net_portrait.pth'))
        self.set_requires_grad(self.u2net, False)

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A_Eyel = input['A']['eyel_A']
        self.real_A_Eyer = input['A']['eyer_A']
        self.real_A_Nose = input['A']['nose_A']
        self.real_A_Mouth = input['A']['mouth_A']
        self.real_A_Hair = input['A']['mask_A']#input['bg']
        self.center = input['A']['center']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.img_A = input['img_A']
        self.landmark_style_5 = input['style_landmark']
        self.landmark_content_5 = input['A']['landmark']
        self.landmark_style_17 = input['landmark_B']
        self.landmark_content_17 = input['landmark_A']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        IMAGE_SIZE = 512
        ratio = IMAGE_SIZE/512
        self.fake_Eyel = F.interpolate(self.real_A_Eyel, scale_factor=ratio, mode='bilinear')
        self.fake_Eyel = torch.flip(self.fake_Eyel, [3])
        self.fake_Eyer = F.interpolate(self.real_A_Eyer, scale_factor=ratio, mode='bilinear')
        self.fake_Nose = F.interpolate(self.real_A_Nose, scale_factor=ratio, mode='bilinear')
        self.fake_Mouth = F.interpolate(self.real_A_Mouth, scale_factor=ratio, mode='bilinear')

        self.landmark_5 = self.inverse_norm(
            self.landmark_predict_5.forward_data(self.landmark_content_5, self.landmark_style_5))

        self.landmark_17_dst = self.inverse_norm_17(
            self.landmark_predict_17.forward_data(self.landmark_content_17, self.landmark_style_17)
        )
        self.landmark_17_src = self.inverse_norm_17(self.landmark_content_17)
        self.fake_Hair = (self.normPRED(1-self.u2net(self.img_A)[0])-0.5)*2 #self.netG_Hair(self.real_A_Hair)#  #
        self.fake_Hair = TF.resize(self.fake_Hair, 512).to(self.device)
        # self.img_A = self.fake_Hair
        self.fake_Hair = self.thin_plate_spline_torch(self.fake_Hair, self.landmark_17_src, self.landmark_17_dst).to(self.device)
        # self.img_A = self.thin_plate_spline_torch(self.img_A, self.landmark_17_src, self.landmark_17_dst).to(
        #     self.device)

        save_landamrk_dir = os.path.join(self.opt.results_dir, self.opt.name, '{}_{}'.format(self.opt.phase, self.opt.epoch),'landmark')
        # os.makedirs(save_landamrk_dir, exist_ok=True)
        # self.write_csv_file(os.path.join(save_landamrk_dir,self.image_paths[0].split('/')[-1].replace('.png','_result.txt')), None, np.array(self.landmark[0], dtype=int))
        self.result = self.partCombiner2(self.fake_Eyel, self.fake_Eyer, self.fake_Nose, self.fake_Mouth,
                                         self.fake_Hair)

        # self.fake_Combine =self.netGCombine(self.real_A)
    def norm(self, points_int, width, height):
        """
        将像素点坐标归一化至 -1 ~ 1
        """
        # points_int_clone = torch.from_numpy(points_int).detach().float().to(device)
        x = ((points_int * 2)[..., 0] / (width - 1) - 1)
        y = ((points_int * 2)[..., 1] / (height - 1) - 1)
        return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)
    def thin_plate_spline_torch(self, img, src, dst):
        # img = img.squeeze(0)
        h, w = img.shape[2], img.shape[3]
        coord = torch.tensor([[
            [0., 0.],
            [h, 0.],
            [h, w],
            [0., w]]])
        coord_src = torch.cat([src,coord], 1)
        coord_dst = torch.cat([dst,coord], 1)

        ten_source = self.norm(coord_src, w, h)
        ten_target = self.norm(coord_dst, w, h)

        tps = TPS()
        warped_grid = tps(ten_target[None, ...], ten_source[None, ...], w, h, 'cpu') # 这个输入的位置需要归一化，所以用norm
        ten_wrp = torch.grid_sampler_2d(img, warped_grid, 0, 0, True)
        # new_img_torch = np.array((ten_wrp[0].cpu()))
        return ten_wrp

    def write_csv_file(self,path, head, data):
        try:
            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, dialect='excel', delimiter=' ')

                if head is not None:
                    writer.writerow(head)

                for row in data:
                    writer.writerow(row)

                print("Write a CSV file to path %s Successful." % path)
        except Exception as e:
            print("Write an CSV file to path: %s, Case: %s" % (path, e))
    def partCombiner2(self, eyel, eyer, nose, mouth, hair, comb_op=-1):
        # if comb_op == 0:
        #     # use max pooling, pad black for eyes etc
        #     padvalue = -1
        #     hair = self.masked(hair, mask)
        # else:
        #     # use min pooling, pad white for eyes etc
        #     padvalue = 1
        #     hair = self.addone_with_mask(hair, mask)
        padvalue = 1
        self.EYE_H = 72
        self.EYE_W = 64
        self.NOSE_H = 40
        self.NOSE_W = 48
        self.MOUTH_H = 40
        self.MOUTH_W = 64
        IMAGE_SIZE = 512
        ratio = IMAGE_SIZE / 256
        div_ratio = int(512/IMAGE_SIZE)
        EYE_W = self.EYE_W * ratio
        EYE_H = self.EYE_H * ratio
        NOSE_W = self.NOSE_W * ratio
        NOSE_H = self.NOSE_H * ratio
        MOUTH_W = self.MOUTH_W * ratio
        MOUTH_H = self.MOUTH_H * ratio
        rhs = [EYE_H, EYE_H, NOSE_H, MOUTH_H]
        rws = [EYE_W, EYE_W, NOSE_W, MOUTH_W]
        bs, nc, _, _ = eyel.shape
        eyel_p = torch.ones((bs, nc, IMAGE_SIZE, IMAGE_SIZE))
        eyer_p = torch.ones((bs, nc, IMAGE_SIZE, IMAGE_SIZE))
        nose_p = torch.ones((bs, nc, IMAGE_SIZE, IMAGE_SIZE))
        mouth_p = torch.ones((bs, nc, IMAGE_SIZE, IMAGE_SIZE))
        for i in range(bs):
            if self.opt.need_landmark_predict:
                feats = self.landmark_5[i]
                mouth_x = int((feats[3, 0] + feats[4, 0]) / 2.0)
                mouth_y = int((feats[3, 1] + feats[4, 1]) / 2.0)
                center = torch.IntTensor([[feats[0, 0], feats[0, 1] - 4 * ratio], [feats[1, 0], feats[1, 1] - 4 * ratio],
                                          [feats[2, 0], feats[2, 1] - NOSE_H / 2 + 16 * ratio], [mouth_x, mouth_y]])
            else:
                center = self.center[i]  # x,y

            eyel_p[i, :, int(center[0, 1] / div_ratio - EYE_H / 2):int((center[0, 1] / div_ratio + EYE_H / 2)),
            int(center[0, 0] / div_ratio - EYE_W / 2):int((center[0, 0] / div_ratio + EYE_W / 2))] = eyel[i]
            eyer_p[i, :, int(center[1, 1] / div_ratio - EYE_H / 2):int((center[1, 1] / div_ratio + EYE_H / 2)),
            int(center[1, 0] / div_ratio - EYE_W / 2):int((center[1, 0] / div_ratio + EYE_W / 2))] = eyer[i]
            nose_p[i, :, int(center[2, 1] / div_ratio - NOSE_H / 2):int((center[2, 1] / div_ratio + NOSE_H / 2)),
            int(center[2, 0] / div_ratio - NOSE_W / 2):int((center[2, 0] / div_ratio + NOSE_W / 2))] = nose[i]
            mouth_p[i, :, int(center[3, 1] / div_ratio - MOUTH_H / 2):int((center[3, 1] / div_ratio + MOUTH_H / 2)),
            int(center[3, 0] / div_ratio - MOUTH_W / 2):int((center[3, 0] / div_ratio + MOUTH_W / 2))] = mouth[i]
            for j in range(4):
                hair[i, :, int(center[j, 1]/div_ratio - rhs[j] / 2):int(center[j, 1]/div_ratio + rhs[j] / 2),
                int(center[j, 0]/div_ratio - rws[j] / 2):int(center[j, 0]/div_ratio + rws[j] / 2)] = 1

        if comb_op == 0:
            # use max pooling
            eyes = torch.max(eyel_p, eyer_p)
            eye_nose = torch.max(eyes, nose_p)
            eye_nose_mouth = torch.max(eye_nose, mouth_p)
            result = torch.max(hair, eye_nose_mouth)
        else:
            # use min pooling
            eyes = torch.min(eyel_p, eyer_p)
            eye_nose = torch.min(eyes, nose_p)
            eye_nose_mouth = torch.min(eye_nose, mouth_p)

            result = torch.min(hair, eye_nose_mouth)
        return result

    def inverse_norm(self, landmark):
        # minx = 117
        # miny = 161
        # maxx = 394
        # maxy = 423
        minx = 100
        maxx = 370
        miny = 140
        maxy = 450
        landmark = landmark.reshape(landmark.shape[0], 5, 2)
        landmark[:, :, 0] = landmark[:, :, 0] * (maxx - minx) + minx
        landmark[:, :, 1] = landmark[:, :, 1] * (maxy - miny) + miny
        return landmark

    def inverse_norm_17(self, landmark):
        minx = 0
        miny = 0
        maxx = 512
        maxy = 512
        landmark = landmark.reshape(landmark.shape[0], 17, 2)
        landmark[:, :, 0] = landmark[:, :, 0] * (maxx - minx) + minx
        landmark[:, :, 1] = landmark[:, :, 1] * (maxy - miny) + miny
        return landmark

    def backward_D_basic(self, netD, real, fake, flag):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        if not flag:
            # Real
            # pred_real = netD(real,0)
            # loss_D_real = self.criterionGAN(pred_real, True)
            # # Fake
            # pred_fake = netD(fake.detach(),0)
            # loss_D_fake = self.criterionGAN(pred_fake, False)ee
            # Combined loss and calculate gradients

            loss_D = self.calc_dis_loss(netD, fake.detach(), real, 0)
            # loss_D.backward()
            return loss_D
        else:
            # pred_real_pair = netD(real,1)
            # pred_fake_pair = netD(fake.detach(),1)
            # # Real
            # loss_D_real = self.criterionGAN(pred_real_pair, True)
            # # Fake
            # loss_D_fake = self.criterionGAN(pred_fake_pair, False)
            loss_D = self.calc_dis_loss(netD, fake.detach(), real, 1)
            # loss_D.backward()
            return loss_D
        # return loss_D

    def backward_D_basic_flur(self, netD, real, fake, flag):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        if not flag:
            # Real
            loss_D = self.calc_dis_loss(netD, fake.detach(), real, 0)
            # loss_D_real = self.criterionGAN(pred_real, True)
            # # Fake
            # pred_fake = netD(fake.detach(),0)
            # loss_D_fake = self.criterionGAN(pred_fake, False)
            # # Combined loss and calculate gradients
            # loss_D = loss_D_real + loss_D_fake
            # loss_D.backward()
            return loss_D
        else:
            # pred_real_pair = netD(real,1)
            # pred_fake_pair = netD(fake.detach(),1)
            # pred_flur_fake = netD(self.flur_B, 1)
            # # Real
            # loss_D_real = self.criterionGAN(pred_real_pair, True)
            # # Fake
            # loss_D_fake = self.criterionGAN(pred_fake_pair, False)
            # loss_D_fake =

            loss_D = self.calc_dis_flur_loss(netD, fake.detach(), real, self.flur_B, 1)
            # loss_D.backward()
            return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_pair_B = self.fake_B_pair_pool.query(self.fake_pair_B)
        self.loss_D_A = self.backward_D_basic_flur(self.netD_A, self.real_B, fake_B, 0)
        self.loss_D_A_pair = self.backward_D_basic_flur(self.netD_A, self.real_B, fake_pair_B, 1)
        loss = self.loss_D_A + self.loss_D_A_pair
        loss.backward()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        fake_pair_A = self.fake_A_pair_pool.query(self.fake_pair_A)
        hed_fake_B = self.hed_fake_B_pool.query(self.hed_fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, 0)
        self.loss_D_B_pair = self.backward_D_basic(self.netD_B, self.real_A, fake_pair_A, 1)

        self.loss_D_hed_B = self.backward_D_basic(self.netD_hed, self.hed_real_B, hed_fake_B,
                                                  0) + self.backward_D_basic(self.netD_hed, self.hed_real_B, hed_fake_B,
                                                                             1)

        loss = self.loss_D_B + self.loss_D_B_pair + self.loss_D_hed_B * self.opt.lambda_hedgan
        loss.backward()

    def compute_dist(self):
        feat_size = 5
        dist_source = torch.zeros(
            [self.opt.batch_size * 2, self.opt.batch_size * 2 - 1, feat_size]).cuda()
        dist_target = torch.zeros(
            [self.opt.batch_size * 2, self.opt.batch_size * 2 - 1, feat_size]).cuda()
        angel_source = torch.zeros(
            [self.opt.batch_size * 2, self.opt.batch_size * 2 - 1, feat_size]).cuda()
        angel_target = torch.zeros(
            [self.opt.batch_size * 2, self.opt.batch_size * 2 - 1, feat_size]).cuda()
        # for i in range(5):
        # cos_value[i] = 1 - self.sim(
        #     (self.real_pair_A_feature[4][2] - torch.mean(self.real_pair_A_feature[4], 0)).reshape(1, -1),
        #     (self.real_pair_B_feature[4][i] - torch.mean(self.real_pair_B_feature[4], 0)).reshape(1, -1))
        # -torch.log(torch.relu(-(cos_value - 1)) + 0.001)
        # dist_pair_target = torch.zeros(
        #     [self.opt.batch_size, self.opt.batch_size - 1, feat_size]).cuda()
        # iterating over different elements in the batch
        # feat_ind = np.random.randint(1, )
        alpha = 0.60
        new_center_A = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.A_feature]
        new_center_B = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.B_feature]
        update_center_A = [alpha * A + (1.0 - alpha) * B for A, B in zip(self.center_A, new_center_A)]
        update_center_B = [alpha * A + (1.0 - alpha) * B for A, B in zip(self.center_B, new_center_B)]
        feat_size = 5
        with torch.set_grad_enabled(False):
            for feat_index in range(5):
                self.A_feature[feat_index] = self.A_feature[feat_index].reshape(self.opt.batch_size * 2, -1)
                self.B_feature[feat_index] = self.B_feature[feat_index].reshape(self.opt.batch_size * 2, -1)
                # A_feature = self.A_feature[feat_index].reshape(self.A_feature[feat_index].shape[0], -1)
                # center_feat = torch.mean(self.A_feature[feat_index], dim=0)
                # next_feat = self.A_feature[feat_index]
                for i in range(1, self.opt.batch_size * 2):
                    next_feat = torch.cat([self.A_feature[feat_index][i:], self.A_feature[feat_index][:i]], 0)
                    dist_source[:, i - 1, feat_index] = self.sim(self.A_feature[feat_index], next_feat)

                    angel_source[:, i - 1, feat_index] = self.sim(
                        self.A_feature[feat_index] - update_center_A[feat_index].reshape(1, -1),
                        next_feat - update_center_A[feat_index].reshape(1, -1))
                # for pair1 in range(self.opt.batch_size * 2):
                #     tmpc = 0
                #     # comparing the possible pairs
                #     for pair2 in range(self.opt.batch_size * 2):
                #         if pair1 != pair2:
                #             anchor_feat = torch.unsqueeze(self.A_feature[feat_index][pair1], 0)
                #             compare_feat = torch.unsqueeze(self.A_feature[feat_index][pair2],0)
                #             dist_source[pair1, tmpc, feat_index] = self.sim(anchor_feat, compare_feat)
                #
                #             # center loss
                #             anchor_feat_vector = torch.unsqueeze((self.A_feature[feat_index][pair1] - center_feat),0)
                #             compare_feat_vector = torch.unsqueeze((self.A_feature[feat_index][pair2] - center_feat),0)
                #             angel_source[pair1, tmpc, feat_index] = self.sim(anchor_feat_vector, compare_feat_vector)
                #             tmpc += 1
            dist_source = self.sfm(dist_source)
            angel_source = self.sfm(angel_source)
        for feat_index in range(5):
            # center_feat = torch.mean(self.B_feature[feat_index], dim=0)
            for pair1 in range(self.opt.batch_size * 2):
                tmpc = 0
                # comparing the possible pairs
                anchor_feat = torch.unsqueeze(self.B_feature[feat_index][pair1], 0)
                anchor_feat_vector = torch.unsqueeze(
                    (self.B_feature[feat_index][pair1] - update_center_B[feat_index].reshape(-1)), 0)
                for pair2 in list(range(pair1, self.opt.batch_size * 2)) + list(range(pair1)):
                    if pair1 != pair2:
                        compare_feat = torch.unsqueeze(self.B_feature[feat_index][pair2], 0)
                        dist_target[pair1, tmpc, feat_index] = self.sim(anchor_feat, compare_feat)

                        # center loss
                        compare_feat_vector = torch.unsqueeze(
                            (self.B_feature[feat_index][pair2] - update_center_B[feat_index].reshape(-1)), 0)
                        angel_target[pair1, tmpc, feat_index] = self.sim(anchor_feat_vector, compare_feat_vector)

                        tmpc += 1
        dist_target = self.sfm(dist_target)
        angel_target = self.sfm(angel_target)
        rel_loss = self.kl(torch.log(dist_target), dist_source)
        center_loss = self.kl(torch.log(angel_target), angel_source)
        # center_loss = 1-torch.mean(self.sim(angel_target, angel_source))
        # center_loss = self.criterionL2(angel_source, angel_target)
        self.center_A = update_center_A
        self.center_B = update_center_B
        # rel_loss = torch.mean(1-self.sim2(dist_source, dist_target))
        return rel_loss, center_loss

    # 到平均的距离,新loss
    def compute_average_dist(self, alpha, beta):
        # beta = 0.1
        new_center_A = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.A_feature]
        # new_pair_center_B = [torch.mean(items, dim=0, keepdim=True) for items in self.real_pair_B_feature]
        new_center_B = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.B_feature]
        # direction_vector = [A - B for A, B in zip(new_center_A, center_A)]
        # 得到单位向量
        # unit_direction_vector = [vec / LA.norm(vec) for vec in direction_vector]

        update_center_A = [alpha * A + (1.0 - alpha) * B for A, B in zip(self.center_A, new_center_A)]
        update_center_B = [alpha * A + (1.0 - alpha) * B for A, B in zip(self.center_B, new_center_B)]
        # update_center_B = [
        #     alpha * A + beta * B + (1.0 - alpha - beta) * unit_vec * rate * LA.norm(dis) for
        #     A, B, unit_vec, dis, rate in
        #     zip(center_B, new_center_B, unit_direction_vector, direction_vector, self.rate)]
        feat_size = 5
        dist_s = torch.zeros([feat_size, self.opt.batch_size * 2]).cuda()
        dist_t = torch.zeros([feat_size, self.opt.batch_size * 2]).cuda()
        avg_dist = torch.zeros([feat_size, 1]).cuda()
        # loss = 0.0
        for i in range(feat_size):
            for j in range(self.opt.batch_size * 2):
                dist_s[i][j] = self.criterionL1(self.A_feature[i][j], update_center_A[i].squeeze(0))
                dist_t[i][j] = self.criterionL1(self.B_feature[i][j], update_center_B[i].squeeze(0))
            avg_dist[i] = torch.mean(dist_t[i])
        self.center_A = update_center_A
        self.center_B = update_center_B
        # loss = torch.mean(dist_t[:,dist_t.shape[1]/2:]-
        # ------------------------------------------------------------
        # normal_dist_s = (dist_s - torch.mean(dist_s, dim=1).unsqueeze(1)) / torch.std(dist_s,dim=1).unsqueeze(1)
        # normal_dist_t = (dist_t - torch.mean(dist_t, dim=1).unsqueeze(1)) / torch.std(dist_t,dim=1).unsqueeze(1)
        # loss = torch.mean(1-self.sim(normal_dist_s, normal_dist_t))
        # ------------------------------------------------------------
        loss = self.kl_dis(torch.log(dist_s / torch.sum(dist_s, dim=1, keepdim=True)),
                           dist_t / torch.sum(dist_s, dim=1, keepdim=True))  # get some remarkable result
        # loss = self.tripletLoss()
        # ------------------------------------------------------------
        # center loss
        # loss = 0.0
        # for i in range(feat_size):
        #     for j in range(self.opt.batch_size * 2):
        #         loss += 0.5*self.criterionL2(self.B_feature[i][j], update_center_B[i].squeeze(0))
        # ------------------------------------------------------------
        # loss = torch.mean(torch.std(dist_t, dim = 1)) #4
        # -------------------
        # -----------------------------------------
        # loss = torch.mean(1-self.sim(dist_s, dist_t)) #3
        # ------------------------------------------------------------
        # loss = F.relu(self.kl_dis(torch.log(dist_t),dist_s))  # get some remarkable result #2
        # ------------------------------------------------------------
        # loss = self.kl_dis(torch.log(dist_t),dist_s)  # get some remarkable result
        # ------------------------------------------------------------
        # loss = self.kl_dis(F.log_softmax(dist_t, dim=1), F.softmax(dist_s, dim=1)) # fix kl_diverge < 0 #
        # ------------------------------------------------------------
        # loss = self.kl_dis(torch.log(self.sfm(dist_s)), self.sfm(dist_t))
        # ------------------------------------------------------------
        # torch.abs(dist_t[i] - avg_dist[i][0]).mean()
        # ------------------------------------------------------------
        # loss = torch.mean(torch.mean(torch.abs(dist_t-avg_dist), dim = 1))
        # ------------------------------------------------------------
        # loss = self.criterionL1(dist_s, dist_t) #1
        # -------------------------------------------------------------
        # loss = self.kl_categorical(dist_t, dist_s)
        # ------------------------------------------------------------
        # torch.var()
        return loss * 0.1

    def sp_module(self):
        feat_size = 5
        loss = 0.0
        for i in range(feat_size):
            loss += self.criterionL2(self.real_A_feature[i], self.fake_A2B_feature[i])
        return loss

    # def centerLoss(self):
    # loss = 0.5 *

    def kl_categorical(self, p_logit, q_logit):
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
        return torch.mean(_kl)

    # 余弦距离（到平均脸的距离）
    def compute_cos_similar(self, feature_A, feature_B):
        loss = 0.0
        avg_sim = []
        sim = torch.zeros(
            [len(feature_A), feature_A[0].shape[0]]).cuda()
        # for i in range(len(feature_A)):
        #     avg_sim.append(torch.mean(1 - self.feature_sim(
        #         (self.real_pair_A_feature[i] - torch.mean(self.real_pair_A_feature[i], 0)).reshape(
        #             self.real_pair_A_feature[i].shape[0], -1),
        #         (self.real_pair_B_feature[i] - torch.mean(self.real_pair_B_feature[i], 0)).reshape(
        #             self.real_pair_B_feature[i].shape[0], -1))))
        for i in range(len(feature_A)):
            sim[i] = 1 - self.feature_sim(
                (feature_A[i] - torch.mean(feature_A[i], 0)).reshape(feature_A[i].shape[0], -1),
                (feature_B[i] - torch.mean(feature_B[i], 0)).reshape(feature_B[i].shape[0], -1))

        return torch.mean(sim)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_context = self.opt.lambda_contextual
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_triplet = self.opt.lambda_triplet
        lambda_rel = self.opt.lambda_rel
        lambda_avg = self.opt.lambda_avg
        lambda_L1 = self.opt.lambda_L1
        lambda_sw = self.opt.lambda_sw
        # hed loss
        ts = self.real_A.shape
        self.hed_real_pair_A = (self.hed(self.real_pair_A / 2 + 0.5) - 0.5) * 2
        self.hed_fake_pair_A = (self.hed(self.fake_pair_A / 2 + 0.5) - 0.5) * 2
        self.hed_real_pair_A = self.hed_real_pair_A.expand(ts)
        self.hed_fake_pair_A = self.hed_fake_pair_A.expand(ts)
        self.loss_L1_pair_A = self.criterionL1(self.hed_real_pair_A, self.hed_fake_pair_A) * lambda_L1
        self.loss_L1_pair_B = self.criterionL1(self.fake_pair_B, self.real_pair_B) * lambda_L1

        # For visual
        self.hed_real_B = (self.hed(self.real_B / 2 + 0.5) - 0.5) * 2
        self.hed_real_B = self.hed_real_B.expand(ts)
        self.hed_fake_B = (self.hed(self.fake_B / 2 + 0.5) - 0.5) * 2
        self.hed_fake_B = self.hed_fake_B.expand(ts)

        # Compute HED loss
        self.hed_A = (self.hed(self.real_A / 2 + 0.5) - 0.5) * 2
        self.hed_rec_A = (self.hed(self.rec_A / 2 + 0.5) - 0.5) * 2
        self.hed_A = self.hed_A.expand(ts)
        self.hed_rec_A = self.hed_rec_A.expand(ts)
        self.loss_hed = self.criterionL1(self.hed_rec_A, self.hed_A) * self.opt.lambda_hed

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            self.loss_identity = self.criterionIdt(self.fake_B, self.real_A) * 0.0
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            self.loss_identity = 0

        self.fake_A2B_feature = self.vgg19(self.fake_B, ['r12', 'r22', 'r32', 'r42', 'r52'])
        self.real_A_feature = self.vgg19(self.real_A, ['r12', 'r22', 'r32', 'r42', 'r52'])
        self.fake_B2A_feature = self.vgg19(self.fake_A, ['r12', 'r22', 'r32', 'r42', 'r52'])
        self.real_B_feature = self.vgg19(self.real_B, ['r12', 'r22', 'r32', 'r42', 'r52'])
        self.real_pair_B_feature = self.vgg19(self.real_pair_B, ['r12', 'r22', 'r32', 'r42', 'r52'])
        self.real_pair_A_feature = self.vgg19(self.real_pair_A, ['r12', 'r22', 'r32', 'r42', 'r52'])
        self.sw = sw_loss.Slicing_torch(self.device, self.real_B_feature, repeat_rate=1)

        self.A_feature = [torch.cat((itemA, itemB), 0) for (itemA, itemB) in
                          zip(self.real_pair_A_feature, self.real_A_feature)]
        self.B_feature = [torch.cat((itemA, itemB), 0) for (itemA, itemB) in
                          zip(self.real_pair_B_feature, self.fake_A2B_feature)]
        self.loss_cos = self.compute_cos_similar(self.real_A_feature, self.fake_A2B_feature) * 1.0
        # {key: np.hstack([self.real_pair_B_feature[key], self.real_B_feature[key]]) for key in
        #               self.real_B_feature.keys()}

        # self.dis_A = [LA.norm(items[0] - items[1]) for items in self.real_pair_A_feature]
        # self.dis_B = [LA.norm(items[0] - items[1]) for items in self.real_pair_B_feature]
        # print("1:{}".format(torch.cuda.memory_allocated(0)))
        if lambda_context > 0.0:
            # fake_A2B_feature = self.vgg19(self.fake_B, ['r12', 'r22', 'r32', 'r42', 'r52'])
            # real_A_feature = self.vgg19(self.real_A, ['r12', 'r22', 'r32', 'r42', 'r52'])
            # fake_B2A_feature = self.vgg19(self.fake_A, ['r12', 'r22', 'r32', 'r42', 'r52'])
            # real_B_feature = self.vgg19(self.real_B, ['r12', 'r22', 'r32', 'r42', 'r52'])
            self.loss_contextual_A = self.ContextualLoss(self.fake_A2B_feature[3:],
                                                         self.real_A_feature[3:]) * lambda_context

            # self.loss_contextual_B = self.ContextualLoss(fake_B2A_feature,real_B_feature) * lambda_context*0.5
        # GAN loss D_A(G_A(A))
        # print("2:{}".format(torch.cuda.memory_allocated(0)))
        self.loss_G_A = self.calc_gen_loss(self.netD_A, self.fake_B, 0)
        self.loss_G_A_pair_fea = self.calc_gen_loss(self.netD_A, self.fake_pair_B, 1)

        # self.loss_G_A_pair_fea2 = self.criterionGAN(D_B_pair_fea[1], True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.calc_gen_loss(self.netD_B, self.fake_A, 0)
        self.loss_G_B_pair_fea = self.calc_gen_loss(self.netD_B, self.fake_pair_A, 1)
        # print("3:{}".format(torch.cuda.memory_allocated(0)))
        self.loss_G_hed_B = self.calc_gen_loss(self.netD_hed, self.hed_fake_B.contiguous(), 0) + self.calc_gen_loss(
            self.netD_hed, self.hed_fake_B.contiguous(), 1)
        # self.loss_G_B_pair_fea2 = self.criterionGAN(D_A_pair_fea[1], True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # Compute L1 loss for paired data
        # self.loss_L1_pair_A = self.criterionL1(self.fake_pair_A, self.real_pair_A) * self.opt.lambda_L1 * 0.5

        # combined loss and calculate gradients
        #  self.loss_cycle_A
        # triplrt loss
        self.loss_triplet = 0.0  # self.tripletLoss(self.hed_fake_B, self.hed_real_B, self.hed_A) * lambda_triplet
        if (self.rate == 0.0):
            self.center_A = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.real_pair_A_feature]
            self.center_B = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.real_pair_B_feature]
        # while (self.rate == 0.0):
        #     import random
        #     random.seed()
        #     random_a = random.randint(0, self.opt.batch_size - 1)
        #     random_b = random.randint(0, self.opt.batch_size - 1)
        #     while (self.real_pair_A_feature[0][random_a].equal(self.real_pair_B_feature[0][random_b])):
        #         random_a = random.randint(0, self.opt.batch_size - 1)
        #         random_b = random.randint(0, self.opt.batch_size - 1)
        #     self.rate = [LA.norm(itemsB[random_a] - itemsB[random_b]) / LA.norm(itemsA[random_a] - itemsA[random_b]) for
        #                  itemsA, itemsB in
        #                  zip(self.real_pair_A_feature, self.real_pair_B_feature)]
        # tv loss
        self.loss_tv = 0.0  # self.tv_loss(self.fake_B) * 0.0
        # print("4:{}".format(torch.cuda.memory_allocated(0)))
        # self.loss_avg_dis = 0.0#self.compute_average_dist(0.7, 0.2)
        # print("5:{}".format(torch.cuda.memory_allocated(0)))
        # if self.opt.lambda_rel > 0.0:
        self.loss_rel, self.loss_avg_dis = self.compute_dist()
        self.loss_rel = self.loss_rel * lambda_rel
        self.loss_avg_dis = self.loss_avg_dis * lambda_avg
        # self.loss_G += self.loss_rel
        self.loss_sw = self.sw(self.fake_A2B_feature) * lambda_sw
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B \
                      + self.loss_identity + self.loss_L1_pair_A + self.loss_L1_pair_B + self.loss_hed \
                      + self.loss_G_A_pair_fea + self.loss_G_B_pair_fea + self.loss_triplet + self.loss_tv \
                      + self.loss_cos + self.loss_rel + self.loss_avg_dis + self.loss_sw + self.loss_G_hed_B * self.opt.lambda_hedgan
        if self.opt.lambda_contextual > 0.0:
            self.loss_G += self.loss_contextual_A
        # with torch.autograd.detect_anomaly():
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_hed],
                               False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_hed], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def calc_dis_loss(self, netD, input_fake, input_real, flag):
        # calculate the loss to train D
        outs0 = netD(input_fake, flag)
        outs1 = netD(input_real, flag)
        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.opt.gan_mode == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.opt.gan_mode == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                  F.binary_cross_entropy(F.sigmoid(out1), all1)) + loss
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, netD, input_fake, flag):
        # calculate the loss to train G
        outs0 = netD(input_fake, flag)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.opt.gan_mode == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.opt.gan_mode == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1)) + loss
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_dis_flur_loss(self, net_D, input_fake, input_real, input_flur, flag):
        # calculate the loss to train D
        outs0 = net_D(input_fake, flag)
        outs1 = net_D(input_real, flag)
        outs2 = net_D(input_flur, flag)
        loss = 0
        for it, (out0, out1, out2) in enumerate(zip(outs0, outs1, outs2)):
            if self.opt.gan_mode == 'lsgan':
                loss += (torch.mean((out0 - 0) ** 2) + torch.mean((out2 - 0) ** 2)) / 2.0 + torch.mean((out1 - 1) ** 2)
            elif self.opt.gan_mode == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                  F.binary_cross_entropy(F.sigmoid(out1), all1)) + loss
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss
