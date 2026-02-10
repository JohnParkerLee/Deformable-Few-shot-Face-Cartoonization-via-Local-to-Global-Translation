import itertools

import torch
import torch.nn.functional as F
from torch import linalg as LA
from torch.autograd import Variable
from . import dist_model as dm
from util.image_pool import ImagePool
from . import networks
from .base_model import BaseModel
from .vgg import ContextualLoss_forward
from .vgg import VGG19
from . import sw_loss


def truncate(fake_B, a=127.5):  # [-1,1]
    return ((fake_B + 1) * a).int().float() / a - 1


class WocatUnpairOriCycleGANModel(BaseModel):
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
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--hed_pretrained_mode', type=str, default='./checkpoints/network-bsds500.pytorch',
                                help='path to the pretrained hed model')
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=15.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_contextual', type=float, default=0.0,
                                help='content loss')
            parser.add_argument('--lambda_L1', type=float, default=10.0)
            parser.add_argument('--lambda_hed', type=float, default=10.0)
            parser.add_argument('--lambda_triplet', type=float, default=0.0)
            parser.add_argument('--lambda_avg', type=float, default=100.0)
            parser.add_argument('--lambda_sw', type=float, default=1e-5)
            parser.add_argument('--lambda_hedgan', type=float, default=1.0)
            parser.add_argument('--lambda_trunc', type=float, default=5.0)
            parser.add_argument('--trunc_a', type=float, default=31.875,
                                help='multiply which value to round when trunc')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'hed', 'idt_A', 'D_B', 'D_hed_B', 'G_B', 'G_hed_B', 'cycle_B', 'idt_B',
                           'identity', 'cos', 'avg_dis', 'sw']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            visual_names_A = ['real_A', 'fake_B', 'rec_A', 'hed_real_A', 'hed_rec_A']
            visual_names_B = ['real_B', 'fake_A', 'rec_B', 'flur_B', 'hed_real_B',
                              'hed_fake_B']
        else:
            visual_names_A = ['real_A', 'fake_B', 'rec_A']
            visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
            visual_names_A.append('idt_fakeB')
        if self.isTrain and self.opt.lambda_trunc > 0.0:
            self.loss_names.append('trunc')

        if self.isTrain:
            vgg_dict_path = './weight/vgg19_conv.pth'
            self.vgg19 = VGG19().cuda()
            self.vgg19.load_state_dict(torch.load(vgg_dict_path))
            self.vgg19.eval()
            if self.opt.lambda_contextual > 0.0:
                self.ContextualLoss = ContextualLoss_forward(h=0.5)
                self.loss_names.append('contextual_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_hed']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_hed = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A_pair_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pair_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.hed_fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.tripletLoss = torch.nn.TripletMarginLoss(margin=0.2, p=1)
            self.sfm = torch.nn.Softmax(dim=1)
            self.feature_sim = torch.nn.CosineSimilarity()
            self.sim2 = torch.nn.CosineSimilarity(dim=2)
            self.criterionL2 = torch.nn.MSELoss()
            self.kl = torch.nn.KLDivLoss()
            self.kl_dis = torch.nn.KLDivLoss()
            self.sim = torch.nn.CosineSimilarity()
            self.tv_loss = networks.TVLoss()
            self.flag = 0
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_hed.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.hed = networks.define_HED(init_weights_=opt.hed_pretrained_mode, gpu_ids_=self.opt.gpu_ids)
            self.lpips = dm.DistModel(opt, model='net-lin', net='alex', use_gpu=True)
            self.set_requires_grad(self.hed, False)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        if self.isTrain:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.flur_B = input['flur_B' if AtoB else 'flur_A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
        else:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)      A--->B
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A)) A--->B--->A
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)      B--->A
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B)) B--->A--->B

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
            loss_D = self.calc_dis_loss(netD, fake.detach(), real, 0)
            return loss_D
        else:
            loss_D = self.calc_dis_loss(netD, fake.detach(), real, 1)
            return loss_D
        # return loss_D

    def backward_D_basic_flur(self, netD, real, fake, flag):
        """Calculate GAN loss for the discriminator"""
        if not flag:
            loss_D = self.calc_dis_loss(netD, fake.detach(), real, 0)
            return loss_D
        else:
            loss_D = self.calc_dis_flur_loss(netD, fake.detach(), real, self.flur_B, 1)
            return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic_flur(self.netD_A, self.real_B, fake_B, 1) + self.backward_D_basic_flur(
            self.netD_A, self.real_B, fake_B, 0)

        loss = self.loss_D_A
        loss.backward()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        hed_fake_B = self.hed_fake_B_pool.query(self.hed_fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, 0) + self.backward_D_basic(self.netD_B,
                                                                                                           self.real_A,
                                                                                                           fake_A, 1)
        self.loss_D_hed_B = self.backward_D_basic(self.netD_hed, self.hed_real_B, hed_fake_B,
                                                  0) + self.backward_D_basic(self.netD_hed, self.hed_real_B, hed_fake_B,
                                                                             1)
        loss = self.loss_D_B + self.loss_D_hed_B * self.opt.lambda_hedgan
        loss.backward()

    def compute_dist(self):
        feat_size = 5
        angel_source = torch.zeros(
            [self.opt.batch_size, self.opt.batch_size - 1, feat_size]).cuda()
        angel_target = torch.zeros(
            [self.opt.batch_size, self.opt.batch_size - 1, feat_size]).cuda()
        adaptive_alpha = torch.mean(
            self.sim(self.real_B.reshape(self.opt.batch_size, -1),
                     self.fake_B.reshape(self.opt.batch_size, -1))).detach()
        if adaptive_alpha >= 0.8:
            adaptive_alpha = 0.5
        alpha = 0.50
        new_center_A = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.A_feature]
        new_center_B = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.B_feature]
        update_center_A = [alpha * A + (1.0 - alpha) * B for A, B in zip(self.center_A, new_center_A)]
        update_center_B = [(1.0 - adaptive_alpha) * A + adaptive_alpha * B for A, B in zip(self.center_B, new_center_B)]
        with torch.set_grad_enabled(False):
            for feat_index in range(feat_size):
                self.A_feature[feat_index] = self.A_feature[feat_index].reshape(self.opt.batch_size, -1)
                self.B_feature[feat_index] = self.B_feature[feat_index].reshape(self.opt.batch_size, -1)
                for pair1 in range(self.opt.batch_size):
                    tmpc = 0
                    # comparing the possible pairs
                    for pair2 in range(self.opt.batch_size):
                        anchor_feat_vector = torch.unsqueeze(
                            (self.A_feature[feat_index][pair1] - update_center_A[feat_index].reshape(-1)), 0)
                        if pair1 != pair2:
                            # center loss
                            compare_feat_vector = torch.unsqueeze(
                                (self.A_feature[feat_index][pair2] - update_center_A[feat_index].reshape(-1)), 0)
                            angel_source[pair1, tmpc, feat_index] = self.sim(anchor_feat_vector, compare_feat_vector)
                            tmpc += 1
            angel_source = self.sfm(angel_source)

        for feat_index in range(feat_size):
            for pair1 in range(self.opt.batch_size):
                tmpc = 0
                # comparing the possible pairs
                anchor_feat_vector = torch.unsqueeze(
                    (self.B_feature[feat_index][pair1] - update_center_B[feat_index].reshape(-1)), 0)
                for pair2 in range(self.opt.batch_size):
                    if pair1 != pair2:
                        # center loss
                        compare_feat_vector = torch.unsqueeze(
                            (self.B_feature[feat_index][pair2] - update_center_B[feat_index].reshape(-1)), 0)
                        angel_target[pair1, tmpc, feat_index] = self.sim(anchor_feat_vector, compare_feat_vector)

                        tmpc += 1
        angel_target = self.sfm(angel_target)
        center_loss = self.kl(torch.log(angel_target), angel_source)
        self.center_A = update_center_A
        self.center_B = update_center_B
        return center_loss

    # 到平均的距离,新loss
    def compute_average_dist(self, alpha, beta):
        # beta = 0.1
        new_center_A = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.A_feature]
        new_center_B = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.B_feature]

        update_center_A = [alpha * A + (1.0 - alpha) * B for A, B in zip(self.center_A, new_center_A)]
        update_center_B = [alpha * A + (1.0 - alpha) * B for A, B in zip(self.center_B, new_center_B)]
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
        loss = self.kl_dis(torch.log(dist_s / torch.sum(dist_s, dim=1, keepdim=True)),
                           dist_t / torch.sum(dist_s, dim=1, keepdim=True))  # get some remarkable result
        return loss * 0.1

    def sp_module(self):
        feat_size = 5
        loss = 0.0
        for i in range(feat_size):
            loss += self.criterionL2(self.real_A_feature[i], self.fake_A2B_feature[i])
        return loss

    def kl_categorical(self, p_logit, q_logit):
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
        return torch.mean(_kl)

    # 余弦距离（到平均脸的距离）
    def compute_cos_similar(self, feature_A, feature_B):
        sim = torch.zeros(
            [len(feature_A), feature_A[0].shape[0]]).cuda()

        for i in range(len(feature_A)):
            sim[i] = 1 - self.feature_sim(
                (feature_A[i] - torch.mean(feature_A[i], 0)).reshape(feature_A[i].shape[0], -1),
                (feature_B[i] - torch.mean(feature_B[i], 0)).reshape(feature_B[i].shape[0], -1))
            # sim[i] = 1 - self.feature_sim((feature_A[i]-self.center_A[i]).reshape(self.opt.batch_size, -1), (feature_B[i]-self.center_B[i]+(self.center_A[i]-self.center_B[i])).reshape(self.opt.batch_size,-1))

        return torch.mean(sim)

    def compute_identity_similar(self, feature_A, feature_B):
        sim = torch.zeros(
            [len(feature_A), feature_A[0].shape[0]]).cuda()

        for i in range(len(feature_A)):
            sim[i] = self.criterionL2(feature_A[i], feature_B[i])

        return torch.mean(sim)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        trunc_a = self.opt.trunc_a
        lambda_idt = self.opt.lambda_identity
        lambda_context = self.opt.lambda_contextual
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_avg = self.opt.lambda_avg
        lambda_L1 = self.opt.lambda_L1
        lambda_sw = self.opt.lambda_sw
        lambda_trunc = self.opt.lambda_trunc * 0.9 * 0.0
        # hed loss
        ts = self.real_A.shape

        # For visual
        self.hed_real_B = (self.hed(self.real_B / 2 + 0.5) - 0.5) * 2
        self.hed_real_B = self.hed_real_B.expand(ts)
        self.hed_fake_B = (self.hed(self.fake_B / 2 + 0.5) - 0.5) * 2
        self.hed_fake_B = self.hed_fake_B.expand(ts)

        # Compute HED loss
        self.hed_real_A = (self.hed(self.real_A / 2 + 0.5) - 0.5) * 2
        self.hed_rec_A = (self.hed(self.rec_A / 2 + 0.5) - 0.5) * 2
        self.hed_fake_A = (self.hed(self.fake_A / 2 + 0.5) - 0.5) * 2
        self.hed_real_A = self.hed_real_A.expand(ts)
        self.hed_rec_A = self.hed_rec_A.expand(ts)
        self.hed_fake_A = self.hed_fake_A.expand(ts)
        # self.loss_hed = self.criterionL1(self.hed_rec_A, self.hed_real_A) * self.opt.lambda_hed
        self.loss_hed = (self.lpips.forward_pair(self.hed_rec_A, self.hed_real_A).mean()) * self.opt.lambda_hed

        self.idt_A = self.netG_A(self.real_B)
        self.hed_idt_A = (self.hed(self.idt_A / 2 + 0.5) - 0.5) * 2
        self.hed_idt_A = self.hed_idt_A.expand(ts)
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||

            self.idt_fakeB = self.netG_A(self.fake_B)
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
        self.sw = sw_loss.Slicing_torch(self.device, self.real_B_feature, repeat_rate=1)
        if (self.flag == 0):
            self.flag = 1
            self.center_A = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.real_A_feature]
            self.center_B = [torch.mean(items.detach(), dim=0, keepdim=True) for items in self.real_B_feature]

        self.A_feature = [torch.cat(itemA, 0) for (itemA) in
                          zip(self.real_A_feature)]
        self.B_feature = [torch.cat((itemA), 0) for (itemA) in
                          zip(self.fake_A2B_feature)]
        self.lambda_similatity = 1.0

        self.loss_cos = self.compute_cos_similar(self.real_A_feature,
                                                 self.fake_A2B_feature) * self.lambda_similatity + self.compute_cos_similar(
            self.real_B_feature, self.fake_B2A_feature) * self.lambda_similatity

        if lambda_context > 0.0:
            self.loss_contextual_A = self.ContextualLoss(self.fake_A2B_feature[3:],
                                                         self.real_A_feature[3:]) * lambda_context

        self.loss_G_A = self.calc_gen_loss(self.netD_A, self.fake_B, 0) + self.calc_gen_loss(self.netD_A, self.fake_B,
                                                                                             1)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.calc_gen_loss(self.netD_B, self.fake_A, 0) + self.calc_gen_loss(self.netD_B, self.fake_A,
                                                                                             1)

        self.loss_G_hed_B = self.calc_gen_loss(self.netD_hed, self.hed_fake_B.contiguous(), 0) + self.calc_gen_loss(
            self.netD_hed, self.hed_real_B.contiguous(), 1)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # truncation loss
        if lambda_trunc != 0.0:
            self.rec_truncA = self.netG_B(truncate(self.fake_B, trunc_a))
            rec_truncA_hed = (self.hed(self.rec_truncA / 2 + 0.5) - 0.5) * 2
            self.loss_trunc = (self.lpips.forward_pair(rec_truncA_hed.expand(ts),
                                                       self.hed_real_A.expand(ts)).mean()) * lambda_trunc
        else:
            self.loss_trunc = 0.0
        if lambda_avg == 0:
            self.loss_avg_dis = 0.0
        else:
            self.loss_avg_dis = self.compute_dist()
            self.loss_avg_dis = self.loss_avg_dis * lambda_avg

        self.loss_sw = self.sw(self.fake_A2B_feature) * lambda_sw
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B \
                      + self.loss_identity + self.loss_hed \
                      + self.loss_cos  + self.loss_avg_dis + self.loss_sw + self.loss_G_hed_B * self.opt.lambda_hedgan  # + self.loss_newidentity
        if self.opt.lambda_contextual > 0.0:
            self.loss_G += self.loss_contextual_A

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
                loss += (torch.mean((out0 - 0) ** 2) + torch.mean((out2 - 0) ** 2)) / 2.0 + torch.mean(
                    (out1 - 1) ** 2)  # + torch.mean((out2 - 0) ** 2)
            elif self.opt.gan_mode == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                  F.binary_cross_entropy(F.sigmoid(out1), all1)) + loss
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss
