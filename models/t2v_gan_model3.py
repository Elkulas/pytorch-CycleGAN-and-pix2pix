import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import ipdb

import sys
sys.path.append('raft/core')

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

class ReCycleGANModel(BaseModel):
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
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        # 如果是train的话还需要添加撒谎那个identity的loss
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        
        
        BaseModel.__init__(self, opt)

        ## 添加对p的loss更新
        if self.isTrain:
            self.adversarial_loss_p = opt.adversarial_loss_p

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A0', 'fake_B0', 'rec_A']
        visual_names_B = ['real_B0', 'fake_A0', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        if not self.isTrain:
            visual_names_A.append('pred_A0')
            visual_names_A.append('pred_B0')
            visual_names_A.append('sync_A0')
            visual_names_A.append('sync_B0')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'P_A', 'P_B']
        else:  # during test time, only load Gs
            # 目前的推断是仅仅使用Generator的
            self.model_names = ['G_A', 'G_B', 'P_A', 'P_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # 生成所需要的网络，构造网络结构
        # ipdb.set_trace()
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # 生成predict网络
        self.netP_A = networks.define_G(2 * opt.input_nc, opt.input_nc, opt.ngf, opt.netP, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netP_B = networks.define_G(2 * opt.input_nc, opt.input_nc, opt.ngf, opt.netP, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        

        if self.isTrain:  # define discriminators
            # 相比较于之前这边
            # TODO：添加sigmod
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)



        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # 相较于cycle，recycle此处加入网络P的参数
            # QUESTION 这里是一个optimizer优化俩吗
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), 
                                                                self.netP_A.parameters(), self.netP_B.parameters()), 
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.isTrain:
            # 设定raft
            ipdb.set_trace()
            self.raft_model = torch.nn.DataParallel(RAFT(opt))
            self.raft_model.load_state_dict(torch.load(opt.raft_model))
            print("RAFT Load Complete!")
        
        print('---------- Networks initialized -------------')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        ## 使用triplet train
        self.real_A0 = input['A0' if AtoB else 'B0'].to(self.device)
        self.real_A1 = input['A1' if AtoB else 'B1'].to(self.device)
        self.real_A2 = input['A2' if AtoB else 'B2'].to(self.device)

        self.real_B0 = input['B0' if AtoB else 'A0'].to(self.device)
        self.real_B1 = input['B1' if AtoB else 'A1'].to(self.device)
        self.real_B2 = input['B2' if AtoB else 'A2'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        if self.isTrain:
            ### 前向G
            ## 前向G，对A0 A1使用
            self.fake_B0 = self.netG_A(self.real_A0)  # G_A(A0)
            self.fake_B1 = self.netG_A(self.real_A1)  # G_A(A1)
            ## 生成B2的预测
            self.fake_B2 = self.netP_B(torch.cat((self.fake_B0, self.fake_B1), 1))
            ## 前向G, 对B0 B1使用
            self.fake_A0 = self.netG_B(self.real_B0)  # G_B(B0)
            self.fake_A1 = self.netG_B(self.real_B1)  # G_B(B1)
            ## 生成A2的预测
            self.fake_A2 = self.netP_A(torch.cat((self.fake_A0, self.fake_A1), 1))

            ### 前向P
            ## 前向P，对A0 A1使用
            # 用recycle的loss
            self.pred_A2 = self.netP_A(torch.cat((self.real_A0, self.real_A1), 1))
            ## 前向P，对B0 B1使用
            self.pred_B2 = self.netP_B(torch.cat((self.real_B0, self.real_B1), 1))

            ## 有意思，相比较于原始的cycleloss，recycle里面生成的rec是pred出来的
            self.rec_A = self.netG_B(self.fake_B2)   # G_B(P_B(G_A(A)))
            self.rec_B = self.netG_A(self.fake_A2)   # G_A(P_A(G_B(B)))
        else:
            self.fake_B0 = self.netG_A(self.real_A2)
            self.fake_A0 = self.netG_B(self.real_B2)

            self.fake_0 = self.netG_A(self.real_A0);
            self.fake_1 = self.netG_A(self.real_A1);
            # ipdb.set_trace()
            self.pred_B0 = self.netP_A(torch.cat((self.fake_0, self.fake_1), 1))
            self.pred_A0 = self.netP_B(torch.cat((self.real_B0, self.real_B1), 1))
            self.sync_B0 = (self.fake_B0 + self.pred_B0) / 2
            self.sync_A0 = (self.fake_A0 + self.pred_A0) / 2
            self.rec_A = self.netG_B(self.fake_B0)   # G_B(P_B(G_A(A)))
            self.rec_B = self.netG_A(self.fake_A0)   # G_A(P_A(G_B(B)))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        ## 这边由于有三个所以要多写仨
        # QUESTION 这个pool有几把用
        # Answer 就是用来存的
        fake_B0 = self.fake_B_pool.query(self.fake_B0)
        self.loss_D_A0 = self.backward_D_basic(self.netD_A, self.real_B0, fake_B0)

        fake_B1 = self.fake_B_pool.query(self.fake_B1)
        self.loss_D_A1 = self.backward_D_basic(self.netD_A, self.real_B1, fake_B1)

        fake_B2 = self.fake_B_pool.query(self.fake_B2)
        self.loss_D_A2 = self.backward_D_basic(self.netD_A, self.real_B2, fake_B2)

        pred_B = self.fake_B_pool.query(self.pred_B2)
        self.loss_D_A3 = self.backward_D_basic(self.netD_A, self.real_B2, pred_B)

        self.loss_D_A = self.loss_D_A1 + self.loss_D_A2 + self.loss_D_A3 + self.loss_D_A0

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""

        fake_A0 = self.fake_A_pool.query(self.fake_A0)
        self.loss_D_B0 = self.backward_D_basic(self.netD_B, self.real_A0, fake_A0)

        fake_A1 = self.fake_A_pool.query(self.fake_A1)
        self.loss_D_B1 = self.backward_D_basic(self.netD_B, self.real_A1, fake_A1)

        fake_A2 = self.fake_A_pool.query(self.fake_A2)
        self.loss_D_B2 = self.backward_D_basic(self.netD_B, self.real_A2, fake_A2)

        pred_A = self.fake_A_pool.query(self.pred_A2)
        self.loss_D_B3 = self.backward_D_basic(self.netD_B, self.real_B2, pred_A)

        self.loss_D_B = self.loss_D_B1 + self.loss_D_B2 + self.loss_D_B3 + self.loss_D_B0

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        ## 计算GA和GB的loss的位
        
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        # 在recycle里面这边全部置0
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B0)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B0) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A0)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A0) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A0 = self.criterionGAN(self.netD_A(self.fake_B0), True)
        self.loss_G_A1 = self.criterionGAN(self.netD_A(self.fake_B1), True)
        self.loss_G_A2 = self.criterionGAN(self.netD_A(self.fake_B2), True)

        self.loss_G_A = self.loss_G_A0 + self.loss_G_A1 + self.loss_G_A2

        # GAN loss D_B(G_B(B))
        self.loss_G_B0 = self.criterionGAN(self.netD_B(self.fake_A0), True)
        self.loss_G_B1 = self.criterionGAN(self.netD_B(self.fake_A1), True)
        self.loss_G_B2 = self.criterionGAN(self.netD_B(self.fake_A2), True)

        self.loss_G_B = self.loss_G_B0 + self.loss_G_B1 + self.loss_G_B2

        # Prediction loss 本质上还是一个cycle loss
        self.loss_pred_A = self.criterionCycle(self.pred_A2, self.real_A2) * lambda_A
        self.loss_P_A = self.loss_pred_A
        self.loss_pred_B = self.criterionCycle(self.pred_B2, self.real_B2) * lambda_B
        self.loss_P_B = self.loss_pred_B

        ## RECYCLE作者多添加的对于pred网络的强化
        # 本质上就是
        if self.adversarial_loss_p:
            pred_fake = self.netD_B(self.pred_A2)
            self.loss_pred_A_adversarial = self.criterionGAN(pred_fake, True)
            pred_fake = self.netD_A(self.pred_B2)
            self.loss_pred_B_adversarial = self.criterionGAN(pred_fake, True)
        else:
            self.loss_pred_A_adversarial = 0
            self.loss_pred_B_adversarial = 0
        
        ## 前向Cycleloss
        # 这里需要注意这个loss是用的pred出来的来做
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A2) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B2) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = (self.loss_G_A0 + self.loss_G_A1 + self.loss_G_A2 +
                       self.loss_G_B0 + self.loss_G_B1 + self.loss_G_B2 + 
                       self.loss_cycle_A + self.loss_cycle_B + 
                       self.loss_pred_A + self.loss_pred_B + self.loss_pred_A_adversarial + self.loss_pred_B_adversarial + 
                       self.loss_idt_A + self.loss_idt_B)
        self.loss_G.backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # 前向计算所有的情况
        # ipdb.set_trace()
        # ipdb.set_trace()
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
