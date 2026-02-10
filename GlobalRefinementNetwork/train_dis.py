# training process
# 1. 获取psp 512 维度的pre-train model !check
# 2. 获取style image对应的inversion code !check
# 3. 输入真实图像，得到其inversion code，分别输入到fixing stylegan和adaptive stylegan计算center loss，并且判定为false
# 4. 对style image的inversion code做style mixing，将其判定为true


import argparse
import math
import os
import random

import networks_local
import numpy
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
import itertools
try:
    import wandb

except ImportError:
    wandb = None
from model import Generator, Extra
from model import Patch_Discriminator as Discriminator  # , Projection_head
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment
from facenet_pytorch import MTCNN
import dist_model as dm

def calc_gen_loss(netD, input_fake, flag):
    # calculate the loss to train G
    outs0 = netD(input_fake, flag)
    loss = 0
    for it, (out0) in enumerate(outs0):
        loss += torch.mean((out0 - 1) ** 2)  # LSGAN
    return loss

def backward_D_basic(netD, input_fake, input_real ):
    return calc_dis_loss(netD, input_fake, input_real, 0) + calc_dis_loss(netD, input_fake,input_real, 1)

def calc_dis_loss(netD, input_fake, input_real, flag):
    # calculate the loss to train D
    outs0 = netD(input_fake, flag)
    outs1 = netD(input_real, flag)
    loss = 0
    for it, (out0, out1) in enumerate(zip(outs0, outs1)):
        loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
    return loss


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
                (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def get_subspace(args, init_z, vis_flag=False):
    std = args.subspace_std
    bs = args.batch if not vis_flag else args.n_sample
    ind = np.random.randint(0, init_z.size(0), size=bs)
    z = init_z[ind]  # should give a tensor of size [batch_size, 512]
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j], std)
    return z


def get_psp(device='cuda'):
    from argparse import Namespace
    from e4e.models.psp import pSp
    model_path = './models/best_model.pt'
    # model_path = './models/e4e_ffhq_encode.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts, device).eval().to(device)
    return net


# def get_e4e_projection():
#     from argparse import Namespace
#     from e4e.models.psp import pSp
#     model_path = 'models/e4e_ffhq_encode.pt'
#     ckpt = torch.load(model_path, map_location='cpu')
#     opts = ckpt['opts']
#     opts['checkpoint_path'] = model_path
#     opts = Namespace(**opts)
#     net = pSp(opts, device).eval().to(device)
#     return net


@torch.no_grad()
def get_latents(net, x):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    imgs, feat = net.decoder([codes], input_is_latent=True, randomize_noise=False, return_latents=True)
    return imgs, feat


@torch.no_grad()
def projection(img, net, return_latents=True):
    images, w_plus = net(img, randomize_noise=False, return_latents=return_latents)
    return w_plus


def crop_img(img, landmark):
    feats = landmark
    item = {}
    EYE_H = 72
    EYE_W = 64
    NOSE_H = 40
    NOSE_W = 48
    MOUTH_H = 40
    MOUTH_W = 64
    # mouth
    mouth_x = int((feats[3, 0] + feats[4, 0]) / 2.0)
    mouth_y = int((feats[3, 1] + feats[4, 1]) / 2.0)
    regions = ['eyel', 'eyer', 'nose', 'mouth']
    ratio = 1
    EYE_H = int(EYE_H * ratio)
    EYE_W = int(EYE_W * ratio)
    NOSE_H = int(NOSE_H * ratio)
    NOSE_W = int(NOSE_W * ratio)
    MOUTH_H = int(MOUTH_H * ratio)
    MOUTH_W = int(MOUTH_W * ratio)
    center = torch.IntTensor([[feats[0, 0], feats[0, 1] - 4 * ratio], [feats[1, 0], feats[1, 1] - 4 * ratio],
                              [feats[2, 0], feats[2, 1] - NOSE_H / 2 + 16 * ratio], [mouth_x, mouth_y]])
    rhs = [EYE_H, EYE_H, NOSE_H, MOUTH_H]
    rws = [EYE_W, EYE_W, NOSE_W, MOUTH_W]
    for i in range(4):
        item[regions[i]] = img[:, int(center[i, 1] - rhs[i] / 2):int(center[i, 1] + rhs[i] / 2),
                           int(center[i, 0] - rws[i] / 2):int(center[i, 0] + rws[i] / 2)]
        if (regions[i] == 'eyel'):
            item[regions[i]] = torch.flip(item[regions[i]], [2])
    return item[regions[0]].unsqueeze(0), item[regions[1]].unsqueeze(0), item[regions[2]].unsqueeze(0), item[
        regions[3]].unsqueeze(0)


def proj_to_origin(img, net, g_source):
    inverted_style_code = projection(img, net)
    return g_source([inverted_style_code], return_feats=False, randomize_noise=False, input_is_latent=True)[0]


def train(args, loader, generator, discriminator, extra, g_optim, d_optim, e_optim, g_ema,
          device, g_source):
    mtcnn = MTCNN(keep_all=True, device='cuda:0')
    # local discriminator
    # netD_eye = networks_local.define_D(3, 64, 'm_dis', 4, 'none', 'xavier', 0.02, device)
    # netD_nose = networks_local.define_D(3, 64, 'm_dis', 4, 'none', 'xavier', 0.02, device)
    # netD_mouth = networks_local.define_D(3, 64, 'm_dis', 4, 'none', 'xavier', 0.02, device)
    # optimizer_D = torch.optim.Adam(
    #     itertools.chain(netD_eye.parameters(), netD_nose.parameters(), netD_mouth.parameters()),
    #     lr=0.002, betas=(0.5, 0.999))
    netG_Eyer = networks_local.define_G(3,3, 64, 'cyclegan', 'instance',False, 'xavier', 0.02, device, n_downsampling=2)
    netG_Nose = networks_local.define_G(3,3, 64, 'cyclegan', 'instance',False, 'xavier', 0.02, device, n_downsampling=2)
    netG_Mouth = networks_local.define_G(3,3, 64, 'cyclegan', 'instance',False, 'xavier', 0.02, device, n_downsampling=2)
    requires_grad(netG_Eyer, False)
    requires_grad(netG_Nose, False)
    requires_grad(netG_Mouth, False)
    netG_Eyer.load_state_dict(torch.load(args.ckpt_netG_eye, map_location=lambda storage, loc: storage))
    netG_Nose.load_state_dict(torch.load(args.ckpt_netG_nose, map_location=lambda storage, loc: storage))
    netG_Mouth.load_state_dict(torch.load(args.ckpt_netG_mouth, map_location=lambda storage, loc: storage))

    torch.manual_seed(1)
    random.seed(1)

    center_init_source = 0
    center_init_target = 0
    alpha = 0.5
    alphaNew = 0.5
    loader = sample_data(loader)

    imsave_path = os.path.join('samples', args.exp)
    model_path = os.path.join('checkpoints', args.exp)
    inversion_net = get_psp(device)
    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # this defines the anchor points, and when sampling noise close to these, we impose image-level adversarial loss (Eq. 4 in the paper)
    init_z = torch.randn(args.n_train, args.latent, device=device)
    # init_z = torch.load('')

    pbar = range(args.iter)
    sfm = nn.Softmax(dim=1)
    kl_loss = nn.KLDivLoss()
    sim = nn.CosineSimilarity()
    l1_loss = nn.L1Loss()
    lpips = dm.DistModel(args, model='net-lin', net='alex', use_gpu=True)
    # id_loss = IDLoss.IDLoss().to(device).eval()

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    # d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    # g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    g_module = generator
    # d_module = discriminator
    g_ema_module = g_ema.module

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    # this defines which level feature of the discriminator is used to implement the patch-level adversarial loss: could be anything between [0, args.highp] 
    lowp, highp = 0, args.highp

    # the following defines the constant noise used for generating images at different stages of training
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    requires_grad(g_source, False)

    sub_region_z = get_subspace(args, init_z.clone(), vis_flag=True)

    for idx in pbar:
        i = idx + args.start_iter
        which = i % args.subspace_freq  # defines whether we sample from anchor region in this iteration or other

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)
        # sw = sw_loss.Slicing_torch(device, d_source(real_img), repeat_rate=1)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(extra, True)

        if which > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            inverted_style_code = projection(real_img, inversion_net)
            mean_w = generator.get_latent(torch.randn([inverted_style_code.size(0), 512]).to(device)).unsqueeze(
                1).repeat(1,
                          generator.n_latent,
                          1)
            id_swap = list(range(7, generator.n_latent))
            noise = inverted_style_code.clone()
            noise[:, id_swap] = alphaNew * inverted_style_code[:, id_swap] + (1 - alphaNew) * mean_w[:, id_swap]
            del inverted_style_code, mean_w

        if which > 0:
            fake_img, _ = generator(noise)
        else:
            fake_img, _ = generator([noise], input_is_latent=True)

        if args.augment:
            real_img, gc = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(
            fake_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        real_pred, _ = discriminator(
            real_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp), real=True)
        # _, _, points_5_fake = mtcnn.detect(fake_img.permute(0, 2, 3, 1) * 255.0, landmarks=True)
        # _, _, points_5_real = mtcnn.detect(real_img.permute(0, 2, 3, 1) * 255.0, landmarks=True)
        # crop
        # crop_fake = crop_img(fake_img[0], points_5_fake[0].squeeze(0))
        # crop_real = crop_img(real_img[0], points_5_real[0].squeeze(0))
        # for idx, item in enumerate(points_5_fake):
        #     if idx == 0:
        #         continue
        #     else:
        #         res = crop_img(fake_img[idx], points_5_fake[idx].squeeze(0))
        #         crop_fake = [torch.cat((itemA, itemB), 0) for (itemA, itemB) in zip(crop_fake, res)]
        #         res = crop_img(real_img[idx], points_5_real[idx].squeeze(0))
        #         crop_real = [torch.cat((itemA, itemB), 0) for (itemA, itemB) in zip(crop_real, res)]
        #
        # loss_D_local_eyel = backward_D_basic(netD_eye, crop_fake[0].detach(), crop_real[0])
        # loss_D_local_eyer = backward_D_basic(netD_eye, crop_fake[0].detach(), crop_real[0])
        # loss_D_local_nose = backward_D_basic(netD_nose, crop_fake[0].detach(), crop_real[0])
        # loss_D_local_mouth =backward_D_basic(netD_mouth, crop_fake[0].detach(), crop_real[0])
        # loss_D_local = (loss_D_local_eyel+loss_D_local_eyer)/2.0 + loss_D_local_nose + loss_D_local_mouth
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()

        extra.zero_grad()
        # netD_eye.zero_grad()
        # netD_nose.zero_grad()
        # netD_mouth.zero_grad()
        d_loss.backward()
        # loss_D_local.backward()
        d_optim.step()
        e_optim.step()
        # optimizer_D.step()

        if args.augment and args.augment_p == 0:
            ada_augment += torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment = reduce_sum(ada_augment)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred, _ = discriminator(
                real_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
            real_pred = real_pred.view(real_img.size(0), -1)
            real_pred = real_pred.mean(dim=1).unsqueeze(1)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            extra.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim.step()
            e_optim.step()
        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        # requires_grad(netD_eye, False)
        # requires_grad(netD_nose, False)
        # requires_grad(netD_mouth, False)
        requires_grad(extra, False)

        if which > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            # noise = [get_subspace(args, init_z.clone())]
            inverted_style_code = projection(real_img, inversion_net)
            mean_w = generator.get_latent(torch.randn([inverted_style_code.size(0), 512]).to(device)).unsqueeze(
                1).repeat(1,
                          generator.n_latent,
                          1)
            id_swap = list(range(7, generator.n_latent))
            noise = inverted_style_code.clone()
            noise[:, id_swap] = alphaNew * inverted_style_code[:, id_swap] + (1 - alphaNew) * mean_w[:, id_swap]
            del inverted_style_code, mean_w

        if which > 0:
            fake_img, _ = generator(noise)
        else:
            fake_img, _ = generator([noise], input_is_latent=True)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p, gc)
            # real_img, _ = augment(real_img, ada_aug_p, gc)

        fake_pred, _ = discriminator(
            fake_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))

        # adversarial loss
        # loss_G_local_eyel = calc_gen_loss(netD_eye, crop_fake[0], 0)
        # loss_G_local_eyer = calc_gen_loss(netD_eye, crop_fake[1], 0)
        # loss_G_local_nose = calc_gen_loss(netD_nose, crop_fake[2], 0)
        # loss_G_local_mouth = calc_gen_loss(netD_mouth, crop_fake[3], 0)
        # loss_G_local = (loss_G_local_eyel + loss_G_local_eyer)/2.0 + loss_G_local_nose + loss_G_local_mouth

        # l1




        g_loss = g_nonsaturating_loss(fake_pred)



        # distance consistency loss
        # import random
        # fake_pair_img, feat_pair_target = generator([noise], return_feats=True, input_is_latent=True)
        # if which<=0:
        #     # lpips_loss = lpips.forward_pair(real_img, fake_img).mean().to(device)
        #     fake_feat = d_source(fake_img)
        #     with torch.no_grad():
        #         real_feat = d_source(real_img)
        #     lpips_loss = sum(
        #             [F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]
        #         ) / len(fake_feat)
        # else:

        # slice_wass_loss = sw(d_source(fake_img))*1e-5
        # lpips_loss = 0
        with torch.set_grad_enabled(False):
            noise = projection(real_img, inversion_net)
            _, feat_pair_source = g_source([noise], return_feats=True, randomize_noise=False, input_is_latent=True)
            z = torch.randn(args.batch, args.latent, device=device)
            feat_ind = numpy.random.randint(1, g_source.module.n_latent - 1, size=args.feat_const_batch)
            #     # computing source distances
            source_sampled_img, feat_source = g_source([z], return_feats=True)
            feat_source = [torch.cat((itemA, itemB), 0) for (itemA, itemB) in zip(feat_source, feat_pair_source)]
            #     # source center init
            if center_init_source == 0:
                center_init_source = 1
                # center_source = []
                center_source = [torch.mean(feat_source[i].detach(), dim=0).reshape(-1) for i in
                                 range(1, g_source.module.n_latent - 1)]
            else:
                center_source_new = [torch.mean(feat_source[i].detach(), dim=0).reshape(-1) for i in
                                     range(1, g_source.module.n_latent - 1)]
                center_source = [alpha * A + (1.0 - alpha) * B for A, B in zip(center_source, center_source_new)]
            #     # dist_source = torch.zeros(
            #     #     [args.feat_const_batch, args.feat_const_batch - 1]).cuda()
            dist_center_source = torch.zeros(
                [args.feat_const_batch, args.feat_const_batch - 1]).cuda()
            #     # iterating over different elements in the batch
            #     sim_scale_feat_source = []
            #     sim_scale_feat_target = []
            for pair1 in range(args.feat_const_batch):
                tmpc = 0
                anchor_feat = torch.unsqueeze(
                    feat_source[feat_ind[pair1]][pair1].reshape(-1), 0)
                # sim_between_same_fea = []
                # comparing the possible pairs
                for pair2 in range(args.feat_const_batch):
                    if pair1 != pair2:
                        compare_feat = torch.unsqueeze(
                            feat_source[feat_ind[pair1]][pair2].reshape(-1), 0)
                        # dist_source[pair1, tmpc] = sim(
                        #     anchor_feat, compare_feat)
                        # computer center loss
                        dist_center_source[pair1, tmpc] = sim(anchor_feat - center_source[feat_ind[pair1] - 1],
                                                              compare_feat - center_source[feat_ind[pair1] - 1])
                        # if pair2>=args.feat_const_batch/2 or pair1>=args.feat_const_batch/2:
                        #     print(pair1, pair2, '\n')
                        #     sim_between_same_fea.append(dist_center_source[pair1, tmpc])
                        tmpc += 1
                # sim_scale_feat_source.append(torch.mean(torch.stack(sim_between_same_fea)))

            #     # dist_source = sfm(dist_source)
            dist_center_source = sfm(dist_center_source)

        #
        # # computing distances among target generations
        _, feat_pair_target = generator([noise], return_feats=True, randomize_noise=False, input_is_latent=True)
        target_sampled_img, feat_target = generator([z], return_feats=True)
        del noise, z
        feat_target = [torch.cat((itemA, itemB), 0) for (itemA, itemB) in
                       zip(feat_target, feat_pair_target)]
        # # dist_target = torch.zeros(
        # #     [args.feat_const_batch, args.feat_const_batch - 1]).cuda()
        dist_center_target = torch.zeros(
            [args.feat_const_batch, args.feat_const_batch - 1]).cuda()
        # target center init
        if center_init_target == 0:
            center_init_target = 1
            # center_target = []
            center_target = [torch.mean(feat_target[i].detach(), dim=0).reshape(-1) for i in
                             range(1, g_source.module.n_latent - 1)]
        else:
            center_target_new = [torch.mean(feat_target[i].detach(), dim=0).reshape(-1) for i in
                                 range(1, g_source.module.n_latent - 1)]
            # adaptive_alpha = torch.mean(
            #     sim(real_img.reshape(args.feat_const_batch, -1), fake_img.reshape(args.feat_const_batch, -1))).detach()
            adaptive_alpha = 0.5
            if adaptive_alpha >= 0.8:
                adaptive_alpha = 0.5
            center_target = [(1.0 - adaptive_alpha) * A + adaptive_alpha * B for A, B in
                             zip(center_target, center_target_new)]
            # for i in range(1, int(np.ceil(g_source.module.n_latent/2))):
            #     center_target.append(torch.mean(feat_target[i], dim=0).reshape(-1))
        # iterating over different elements in the batch
        for pair1 in range(args.feat_const_batch):
            tmpc = 0
            anchor_feat = torch.unsqueeze(
                feat_target[feat_ind[pair1]][pair1].reshape(-1), 0)
            # sim_between_same_fea = []
            for pair2 in range(args.feat_const_batch):  # comparing the possible pairs
                # pair_feat = torch.unsqueeze(
                #     feat_pair_target[feat_ind[pair1]][pair2].reshape(-1), 0)
                # dist_pair_target[pair2, pair1] = sim(pair_feat - center_target[feat_ind[pair1] - 1],
                #                                      anchor_feat - center_target[feat_ind[pair1] - 1])
                if pair1 != pair2:
                    compare_feat = torch.unsqueeze(
                        feat_target[feat_ind[pair1]][pair2].reshape(-1), 0)
                    # dist_target[pair1, tmpc] = sim(anchor_feat, compare_feat)
                    # computer center loss
                    dist_center_target[pair1, tmpc] = sim(anchor_feat - center_target[feat_ind[pair1] - 1],
                                                          compare_feat - center_target[feat_ind[pair1] - 1])
                    # if pair2>=args.feat_const_batch/2 or pair1>=args.feat_const_batch/2:
                    #     print(pair1, pair2, '\n')
                    #     sim_between_same_fea.append(dist_center_target[pair1, tmpc])
                    tmpc += 1
            # sim_scale_feat_target.append(torch.mean(torch.stack(sim_between_same_fea)))
        # dist_target = sfm(dist_target)
        dist_center_target = sfm(dist_center_target)
        # dist_pair_target = sfm(dist_pair_target)
        # rel_loss = args.kl_wt * \
        #     kl_loss(torch.log(dist_target), dist_source) # distance consistency loss
        center_loss = args.center_wt * kl_loss(torch.log(dist_center_target), dist_center_source)
        # identity_loss = args.identity_wt * id_loss(source_sampled_img, target_sampled_img)
        # center_pair_loss = 90 * kl_loss(torch.log(dist_pair_source), dist_pair_target)

        # crop
        if i%5==0:
#        if False:
            try:
                _, _, points_5_fake = mtcnn.detect(target_sampled_img.permute(0, 2, 3, 1) * 255.0, landmarks=True)
                _, _, points_5_real = mtcnn.detect(source_sampled_img.permute(0, 2, 3, 1) * 255.0, landmarks=True)
            # if len(points_5_fake.shape)>2 and len(points_5_real.shape)>2 and points_5_fake.shape[0]==target_sampled_img.shape[0] and points_5_real.shape[0]==target_sampled_img.shape[0]:
                crop_fake = crop_img(target_sampled_img[0], points_5_fake[0].squeeze(0))
                crop_real = crop_img(source_sampled_img[0], points_5_real[0].squeeze(0))
                for index, item in enumerate(points_5_fake):
                    if index == 0:
                        continue
                    else:
                        res = crop_img(target_sampled_img[index], points_5_fake[index].squeeze(0))
                        crop_fake = [torch.cat((itemA, itemB), 0) for (itemA, itemB) in zip(crop_fake, res)]
                        res = crop_img(source_sampled_img[index], points_5_real[index].squeeze(0))
                        crop_real = [torch.cat((itemA, itemB), 0) for (itemA, itemB) in zip(crop_real, res)]

                local_eyel = netG_Eyer(transforms.functional.resize(crop_real[0], (144, 128)))
                local_eyer = netG_Eyer(transforms.functional.resize(crop_real[1],(144,128)))
                local_nose = netG_Nose(transforms.functional.resize(crop_real[2],(80,96)))
                local_mouth = netG_Mouth(transforms.functional.resize(crop_real[3],(80,128)))
                lpips_eyel = lpips.forward_pair(local_eyel, transforms.functional.resize(crop_fake[0], (144, 128))).mean().to(
                    device)
                lpips_eyer= lpips.forward_pair(local_eyer, transforms.functional.resize(crop_fake[1],(144,128))).mean().to(device)

                lpips_nose = L1_loss(local_nose, transforms.functional.resize(crop_fake[2], (80,96))).mean().to(
                    device)
                lpips_mouse = lpips.forward_pair(local_mouth, transforms.functional.resize(crop_fake[3], (60,128))).mean().to(
                    device)
                lpips_loss = (lpips_eyel + lpips_eyer)/2.0 + lpips_nose + lpips_mouse
            except:
                lpips_loss = 0.0
        else:
            lpips_loss = 0.0






        sim_loss = torch.zeros(
            [args.feat_const_batch, 1]).cuda()
        for pair1 in range(args.feat_const_batch):
            sim_loss[pair1] = 1 - torch.mean(
                sim(feat_source[feat_ind[pair1]].reshape(args.batch * 2, -1) - center_source[feat_ind[pair1] - 1],
                    feat_target[feat_ind[pair1]].reshape(args.batch * 2, -1) - center_target[feat_ind[pair1] - 1]))
        sim_loss = torch.mean(sim_loss) * args.cos_wt

        # lpips loss
        # target_sample,source_sample
        # ts = source_sample.shape
        # (lpips.forward_pair(target_sample.expand(ts), source_sample.expand(ts)).mean()).cuda() * 0.0

        g_loss = g_loss + center_loss + sim_loss + lpips_loss  # + slice_wass_loss# + identity_loss# + center_pair_loss # + pair_l1_loss# rel_loss +

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        # to save up space rel_loss,
        del g_loss, d_loss, fake_img, fake_pred, real_img, real_pred, feat_source, feat_target, dist_center_source, dist_center_target, anchor_feat, compare_feat

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(
                path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema_module, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}; sim: {sim_loss:.4f}; lpips: {lpips_loss:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % args.img_freq == 0:
                with torch.set_grad_enabled(False):
                    g_ema.eval()
                    sample, _ = g_ema([sample_z.data])
                    sample_subz, _ = g_ema([sub_region_z.data])
                    utils.save_image(
                        sample,
                        f"%s/{str(i).zfill(6)}.png" % (imsave_path),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    del sample

            if (i % args.save_freq == 0) and (i > 0):
                torch.save(
                    {
                        "g_ema": g_ema.state_dict(),
                        # uncomment the following lines only if you wish to resume training after saving. Otherwise, saving just the generator is sufficient for evaluations

                        # "g": g_module.state_dict(),
                        # "g_s": g_source.state_dict(),
                        # "d": d_module.state_dict(),
                        # "g_optim": g_optim.state_dict(),
                        # "d_optim": d_optim.state_dict(),
                    },
                    f"%s/{str(i).zfill(6)}.pt" % (model_path),
                )


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--iter", type=int, default=5002)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--img_freq", type=int, default=500)
    parser.add_argument("--kl_wt", type=int, default=1000)
    parser.add_argument('--identity_wt', type=float, default=1.0)
    parser.add_argument('--cos_wt', type=float, default=1.0)
    parser.add_argument('--center_wt', type=float, default=100.0)
    parser.add_argument("--highp", type=int, default=1)
    parser.add_argument("--subspace_freq", type=int, default=4)
    parser.add_argument("--feat_ind", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--feat_const_batch", type=int, default=3)
    parser.add_argument("--n_sample", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--feat_res", type=int, default=128)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--subspace_std", type=float, default=0.1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--inversion_ckpt", type=str, default=None)
    parser.add_argument("--source_key", type=str, default='ffhq')
    parser.add_argument("--exp", type=str, default=None, required=True)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment", dest='augment', action='store_true')
    parser.add_argument("--no-augment", dest='augment', action='store_false')
    parser.add_argument("--augment_p", type=float, default=0.0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--ckpt_netG_eye", type=str, default="/home/ss/work/cycle/checkpoints/sketches_combine/latest_net_G_Eyer.pth")
    parser.add_argument("--ckpt_netG_nose", type=str, default="/home/ss/work/cycle/checkpoints/sketches_combine/latest_net_G_Nose.pth")
    parser.add_argument("--ckpt_netG_mouth", type=str, default="/home/ss/work/cycle/checkpoints/sketches_combine/latest_net_G_Mouth.pth")

    parser.add_argument('--gpu_ids_p', type=str, default='0',
                        help='gpu ids for pretrained auxiliary models: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    args = parser.parse_args()

    n_gpu = 4
    args.distributed = n_gpu > 1

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_source = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    # d_source = Rec_Discriminator(
    #     args.size, channel_multiplier=args.channel_multiplier
    # ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    extra = Extra().to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    e_optim = optim.Adam(
        extra.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    module_source = ['landscapes', 'red_noise',
                     'white_noise', 'hands', 'mountains', 'handsv2']

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        assert args.source_key in args.ckpt
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_source = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g_ema"], strict=False)
        g_source.load_state_dict(ckpt_source["g_ema"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        # d_source = nn.parallel.DataParallel(d_source)
        # discriminator = nn.parallel.DataParallel(discriminator)
        discriminator.load_state_dict(ckpt["d"])
        # d_source.load_state_dict(ckpt_source["d"])

        if 'g_optim' in ckpt.keys():
            g_optim.load_state_dict(ckpt["g_optim"])
        if 'd_optim' in ckpt.keys():
            d_optim.load_state_dict(ckpt["d_optim"])

    if args.inversion_ckpt is not None:
        # from e4e_projection import projection as e4e_projection

        print("load inversion model:", args.inversion_ckpt)

    if args.distributed:
        geneator = nn.parallel.DataParallel(generator)
        g_ema = nn.parallel.DataParallel(g_ema)
        g_source = nn.parallel.DataParallel(g_source)

        discriminator = nn.parallel.DataParallel(discriminator)
        # d_source = nn.parallel.DataParallel(d_source)
        extra = nn.parallel.DataParallel(extra)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.data_path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )
    # if args.style_image_path is not None:
    #     latents = []
    #     style_path = args.style_image_path
    #     style_list = os.listdir(style_path)
    #     for name in style_list:
    #         latent = e4e_projection(os.path.join(style_path, name))
    #         latents.append(latent)
    #     latents = torch.stack(latents, 0)

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, generator, discriminator, extra, g_optim,
          d_optim, e_optim, g_ema, device, g_source)
