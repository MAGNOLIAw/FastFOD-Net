from .loss import *
from utils import *
from .networks import define_net
from .base_model import BaseModel
import torch

class REModel(BaseModel):
    """
    Resolution Enhancement Model for diffusion MRI signal reconstruction.

    This model enhances angular resolution of fiber orientation distributions (FODs)
    by learning to inpaint missing or downsampled components from masked inputs.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add model-specific command-line options.

        Args:
            parser (argparse.ArgumentParser): Original parser.
            is_train (bool): If True, adds options specific to training mode.

        Returns:
            argparse.ArgumentParser: Modified parser with RE-specific options.
        """
        parser.add_argument('--conv_type', type=str, default='unet')

        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        """
        Initialize the Resolution Enhancement model.

        Args:
            opt (argparse.Namespace): Parsed training/testing options.
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['R', 'order1', 'order2', 'order3','order4','order5']
        self.model_names = ['inpaint']
        self.opt = opt

        # Initialize the enhancement network
        self.net_inpaint = define_net(
            self.opt.input_nc, 
            opt.output_nc, 
            opt.conv_type, 
            opt.norm, 
            self.opt.init_type, 
            self.opt.init_gain, 
            gpu_ids=self.opt.gpu_ids)

        self.loss = torch.nn.MSELoss()

        if self.opt.isTrain:
            self.optimizer_inpaint = torch.optim.Adam(
                self.net_inpaint.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_inpaint)       

    def set_input(self, input):
        """
        Unpack input data from dataloader.

        Args:
            input (dict): Dictionary with keys 'brain', 'mask', and 'gt'.
        """
        self.brain = input['brain'].to(self.device)               # Masked input FOD
        self.mask = input['mask'][:, 0:1, ...].to(self.device)    # Binary mask
        self.gt = input['gt'].to(self.device)                     # Ground-truth FOD

    def forward(self):
        """
        Run forward pass
        """
        # IF use pseudo slice as input
        self.inpainted = self.net_inpaint(self.brain)

        # Decompose into spherical harmonic orders
        self.out1 = self.inpainted[:,0:1,:,:,:]
        self.out2 = self.inpainted[:,1:6,:,:,:]
        self.out3 = self.inpainted[:,6:15, :, :, :]
        self.out4 = self.inpainted[:,15:28, :, :, :]
        self.out5 = self.inpainted[:,28:45, :, :, :]

        if not self.isTrain:
            self._compute_loss()

        return self.inpainted

    def _compute_loss(self):
        """
        Compute masked MSE loss for each SH order.
        Used in both training and evaluation.
        """
        self.loss_order1 = self.loss(self.mask * self.out1, self.mask * self.gt[:, 0:1, ...])
        self.loss_order2 = self.loss(self.mask * self.out2, self.mask * self.gt[:, 1:6, ...])
        self.loss_order3 = self.loss(self.mask * self.out3, self.mask * self.gt[:, 6:15, ...])
        self.loss_order4 = self.loss(self.mask * self.out4, self.mask * self.gt[:, 15:28, ...])
        self.loss_order5 = self.loss(self.mask * self.out5, self.mask * self.gt[:, 28:45, ...])

        self.loss_R = (
            self.loss_order1 +
            self.loss_order2 +
            self.loss_order3 +
            self.loss_order4 +
            self.loss_order5
        )
    
    def backward_inpaint(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        """
        # calculate reconstruction loss
        self._compute_loss()
        self.loss_R.backward()

    def optimize_parameters(self):
        """
        Update network weights; it will be called in every training iteration.
        """
        self.forward()

        # update optimizer of the inpainting network
        self.optimizer_inpaint.zero_grad()
        self.backward_inpaint()
        self.optimizer_inpaint.step()


