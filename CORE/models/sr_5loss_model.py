from .loss import *
from utils import *
from .networks import define_net
from .base_model import BaseModel
import torch
import nibabel as nib
import os

class SR5LossModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new LGC-specific options.
        """
        parser.add_argument('--lambda_mask', type=float, default=1, help='weight for mask area L1 loss')
        parser.add_argument('--lambda_tissue', type=float, default=1, help='weight for valid tissue L1 loss')
        parser.add_argument('--conv_type', type=str, default='unet')

        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        """
        Initialize this mask Inpaint class.
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['R', 'order1', 'order2', 'order3','order4','order5']
        self.model_names = ['inpaint']
        self.opt = opt

        # define the inpainting network
        self.net_inpaint = define_net(self.opt.input_nc, opt.output_nc, opt.conv_type, opt.norm,
                                      self.opt.init_type, self.opt.init_gain, gpu_ids=self.opt.gpu_ids)

        if self.opt.isTrain:
            # define the loss functions
            # self.loss = L1Loss(weight=opt.lambda_mask)
            self.loss = torch.nn.MSELoss()
            # define the optimizer
            self.optimizer_inpaint = torch.optim.Adam(self.net_inpaint.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_inpaint)
        else:
            # define the loss functions
            # self.loss = L1Loss(weight=opt.lambda_mask)
            self.loss = torch.nn.MSELoss()
            # self.loss1 = L1Loss(weight=1 / 45)
            # self.loss2 = L1Loss(weight=5 / 45)
            # self.loss3 = L1Loss(weight=9 / 45)
            # self.loss4 = L1Loss(weight=13 / 45)
            # self.loss5 = L1Loss(weight=17 / 45)



    def set_input(self, input):
        """
        Read the data of input from dataloader then
        """
        if self.isTrain:
            self.brain = input['brain'].to(self.device)  # get masked brain, i.e. mask areas have value of 0
            self.mask = input['mask'][:,0:1,:,:,:].to(self.device)
            self.gt = input['gt'].to(self.device)  # get original brain

        else:
            self.brain = input['brain'].to(self.device)  # get masked brain, i.e. mask areas have value of 0
            self.mask = input['mask'][:,0:1,:,:,:].to(self.device)
            self.gt = input['gt'].to(self.device)  # get original brain


    def forward(self):
        """
        Run forward pass
        """
        # IF use pseudo slice as input
        self.inpainted = self.net_inpaint(self.brain)
        # IF combine pseudo slice and corresponding mask as input
        # self.input = torch.cat((self.brain, self.mask), dim=1)
        # self.inpainted = self.net_inpaint(self.input)

        self.out1 = self.inpainted[:,0:1,:,:,:]
        self.out2 = self.inpainted[:,1:6,:,:,:]
        self.out3 = self.inpainted[:,6:15, :, :, :]
        self.out4 = self.inpainted[:,15:28, :, :, :]
        self.out5 = self.inpainted[:,28:45, :, :, :]

        if not self.opt.isTrain:
            self.loss_order1 = self.loss(self.mask * self.out1, self.mask * self.gt[:, 0:1, :, :, :])
            self.loss_order2 = self.loss(self.mask * self.out2, self.mask * self.gt[:, 1:6, :, :, :])
            self.loss_order3 = self.loss(self.mask * self.out3, self.mask * self.gt[:, 6:15, :, :, :])
            self.loss_order4 = self.loss(self.mask * self.out4, self.mask * self.gt[:, 15:28, :, :, :])
            self.loss_order5 = self.loss(self.mask * self.out5, self.mask * self.gt[:, 28:45, :, :, :])
            self.loss_R = self.loss_order1 + self.loss_order2 + self.loss_order3 + self.loss_order4 + self.loss_order5
        return self.inpainted

    def backward_inpaint(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # calculate reconstruction loss
        self.loss_order1 = self.loss(self.mask * self.out1, self.mask * self.gt[:, 0:1, :, :, :])
        self.loss_order2 = self.loss(self.mask * self.out2, self.mask * self.gt[:, 1:6, :, :, :])
        self.loss_order3 = self.loss(self.mask * self.out3, self.mask * self.gt[:, 6:15, :, :, :])
        self.loss_order4 = self.loss(self.mask * self.out4, self.mask * self.gt[:, 15:28, :, :, :])
        self.loss_order5 = self.loss(self.mask * self.out5, self.mask * self.gt[:, 28:45, :, :, :])
        self.loss_R = self.loss_order1 + self.loss_order2 + self.loss_order3 + self.loss_order4 + self.loss_order5
        self.loss_R.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()

        # update optimizer of the inpainting network
        self.optimizer_inpaint.zero_grad()
        self.backward_inpaint()
        self.optimizer_inpaint.step()


