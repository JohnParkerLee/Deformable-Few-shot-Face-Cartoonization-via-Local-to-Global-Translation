import torch
import numpy as np
from . import networks_fully_pointNet as networks
from .base_model import BaseModel


class LandmarkGanModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        # parser.set_defaults(where_add='input', nz=0)
        if is_train:
            parser.set_defaults(gan_mode='vanilla', lambda_l1=100.0)

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'G_style','Triplet_Loss', 'G_', 'D_real', 'D_fake']
        self.loss_names = ['G', 'G_L1', 'G_L1_Cycle', 'G_cos', 'G_triplet', 'G_L1_rec']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_A2B', 'fake_B2A', 'real_B', 'rec_A', 'rec_B']
        if opt.isTrain and not self.opt.finetune:
            self.visual_names.append(['real_A2B', 'real_B2A'])
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        self.model_names = ['G_content', 'G_style']
        self.netG_content, self.netG_style = networks.define_G(opt.input_nc, opt.output_nc, opt.k, opt.nz, opt.ngf,
                                                               netG=opt.netG,
                                                               norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout,
                                                               init_type=opt.init_type,
                                                               init_gain=opt.init_gain,
                                                               gpu_ids=self.gpu_ids, where_add=opt.where_add,
                                                               upsample=opt.upsample)
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm,
                                          nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds,
                                          gpu_ids=self.gpu_ids)
        if opt.isTrain:
            if not opt.finetune:
                self.visual_names.append('real_A2B')
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.cosine = torch.nn.CosineSimilarity(dim=2)
            self.tripleLoss = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)

            self.optimizers = []
            self.optimizer_G_content = torch.optim.Adam(self.netG_content.parameters(), lr=opt.lr,
                                                        betas=(opt.beta1, 0.999))
            self.optimizer_G_style = torch.optim.Adam(self.netG_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_content)
            self.optimizers.append(self.optimizer_G_style)
            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        item = self.opt.part_class
        self.real_A = input['data_A'][item].float().to(self.device)
        self.real_B = input['data_B'][item].float().to(self.device)
        if self.opt.isTrain and not self.opt.finetune:
            self.real_A2B = input['data_AB'][item].float().to(self.device)
            self.real_B2A = input['data_BA'][item].float().to(self.device)
        random_k = np.random.randint(1,10)
        self.real_A_style = input['data_A_style'][item][:,:random_k,:,:].float().to(self.device)
        self.real_B_style = input['data_B_style'][item][:,:random_k,:,:].float().to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.A_feature = torch.mean(self.netG_style(self.real_A_style), dim=1)
        self.B_feature = torch.mean(self.netG_style(self.real_B_style), dim=1)

        self.fake_A2B = self.netG_content(self.real_A, self.B_feature)  # G(A)
        self.fake_B2A = self.netG_content(self.real_B, self.A_feature)
        self.fake_A2A = self.netG_content(self.real_A, self.A_feature)
        self.fake_B2B = self.netG_content(self.real_B, self.B_feature)
        self.fake_A2B_feature = self.netG_style(self.fake_A2B)
        self.fake_B2A_feature = self.netG_style(self.fake_B2A)
        self.rec_A = self.netG_content(self.fake_A2B, self.A_feature)
        self.rec_B = self.netG_content(self.fake_B2A, self.B_feature)
        # print(self.fake_A2B.shape, self.fake_B2A.shape)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real, _ = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        '''
        if not self.opt.finetune:
            self.loss_G_L1 = self.criterionL1(self.fake_A2B, self.real_A2B) * self.opt.lambda_L1

            self.loss_G_L1_cycle = self.criterionL1(self.fake_B2A, self.real_A) * self.opt.lambda_L1 * 0.5
        else:
            self.loss_G_L1 = 0.0
            self.loss_G_L1_cycle = 0.0
        # combine loss and calculate gradients
        # self.Triplet_Loss = self.tripleLoss()
        # self.loss_G_fea_L1 = self.criterionL1(self.B_feature, self.A2B_feature) * self.opt.fea_L1
        # self.loss_G_Triplet = self.tripleLoss(self.fake_A2B, torch.mean(self.real_B, dim=1, keepdim = True), self.real_A) * self.opt.triplet
        # self.loss_G_Triplet = self.tripleLoss(self.fake_A2B, torch.mean(self.real_B, 1, keepdim=True), self.real_A) * self.opt.triplet
        self.loss_G_Triplet = 0
        # for i in range(self.real_B.shape[1]):
        #     self.loss_G_Triplet += self.tripleLoss(self.fake_A2B, self.real_B[:,i,:], self.real_A)
        # self.loss_G_Triplet = (self.loss_G_Triplet/self.real_B.shape[1]) * self.opt.triplet
        self.loss_G_Triplet += self.tripleLoss(self.fake_A2B_feature, self.B_feature, self.A_feature)
        self.loss_G_Triplet += self.tripleLoss(self.fake_B2A_feature, self.A_feature, self.B_feature)
        self.loss_G = self.loss_G_L1 + self.loss_G_Triplet * self.opt.triplet + self.loss_G_L1_cycle
        self.loss_G.backward()
        '''
        if not self.opt.finetune:
            self.loss_G_L1 = (self.criterionL1(self.fake_A2B, self.real_A2B)
                              + self.criterionL1(self.fake_B2A, self.real_B2A)) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0
        self.loss_G_L1_rec = (self.criterionL1(self.fake_A2A, self.real_A) + self.criterionL1(self.fake_B2B,
                                                                                              self.real_B)) * self.opt.lambda_L1
        # cycle loss
        self.loss_G_L1_Cycle = (self.criterionL1(self.rec_A, self.real_A)
                                + self.criterionL1(self.rec_B, self.real_B)) * self.opt.lambda_L1

        # triplet
        # print(self.tripleLoss(self.fake_A2B_feature, self.B_feature, self.A_feature))
        self.loss_G_triplet = (self.tripleLoss(self.fake_A2B_feature, self.B_feature, self.A_feature) +
                               self.tripleLoss(self.fake_B2A_feature, self.A_feature,
                                               self.B_feature)) * self.opt.triplet
        # print((self.cosine(self.fake_A2B - torch.mean(self.real_B_style, dim= 1), self.real_A - torch.mean(self.real_A_style, dim= 1)) ).shape)
        self.loss_G_cos = (1 - torch.mean(self.cosine(self.fake_A2B - torch.mean(self.real_B_style, dim=1),
                                                      self.real_A - torch.mean(self.real_A_style, dim=1)))) * 2.5 + \
                          (1 - torch.mean(self.cosine(self.fake_B2A - torch.mean(self.real_A_style, dim=1),
                                                      self.real_B - torch.mean(self.real_B_style, dim=1)))) * 2.5
        self.loss_G = self.loss_G_L1 + self.loss_G_L1_Cycle + self.loss_G_cos + self.loss_G_triplet + self.loss_G_L1_rec
        self.loss_G.backward()

    def k_shot_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['data_A'][self.opt.part_class].float().to(self.device)
        self.real_B_style = input['data_B_style'][self.opt.part_class].float().to(self.device)
        # print(self.real_B.shape)
        # self.real_B_style = input['data_B']
        # self.real_A2B = input['data_AB'].float().to(self.device)

    def k_shot_test(self):
        with torch.no_grad():
            # for i in range(len(self.real_B[0])):
            #     if i == 0:
            #         self.B_feature = torch.unsqueeze(self.netG_style(self.real_B[:, i, :]), 1)
            #     else:
            #         self.B_feature = torch.cat((self.B_feature, torch.unsqueeze(self.netG_style(self.real_B[:, i, :]), 1)), 1)
            # print(self.real_B.shape)
            print(self.netG_style(self.real_B_style).shape)
            self.B_feature = torch.mean(self.netG_style(self.real_B_style), dim=1)
            self.fake_A2B = self.netG_content(self.real_A, self.B_feature)  # G(A)
        return self.real_A, self.real_B_style, self.fake_A2B

    def test(self):
        '''
        with torch.no_grad():
            for i in range(len(self.real_B[0])):
                if i == 0:
                    self.B_feature = torch.unsqueeze(self.netG_style(self.real_B[:, i, :]), 1)
                else:
                    self.B_feature = torch.cat(
                        (self.B_feature, torch.unsqueeze(self.netG_style(self.real_B[:, i, :]), 1)), 1)
            self.B_feature = torch.mean(self.B_feature, 1)
            self.fake_A2B = self.netG_content(self.real_A, self.B_feature)  # G(A)
        return self.real_A, self.real_B, self.fake_A2B
        '''
        with torch.no_grad():
            self.A_feature = torch.mean(self.netG_style(self.real_A_style), dim=1)
            self.B_feature = torch.mean(self.netG_style(self.real_B_style), dim=1)
            # print('data:', self.A_feature[item].shape)
            # print('data:', self.B_feature[item].shape)
            self.fake_A2B = self.netG_content(self.real_A, self.B_feature)  # G(A)
            self.fake_B2A = self.netG_content(self.real_B, self.A_feature)
        return self.real_A, self.real_B, self.fake_A2B, self.fake_B2A

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D()                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights
        # update G
        # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G_style.zero_grad()  # set G's gradients to zero
        self.optimizer_G_content.zero_grad()
        self.backward_G()  # calculate graidents for G
        self.optimizer_G_style.step()  # udpate G's weights
        self.optimizer_G_content.step()
