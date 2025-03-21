import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from .unet_3d import UNet3D
from .unet_3d_v1 import UNet3Dv1
from .unet_3d_v1_c64 import UNet3Dv1c64
from .unet_3d_dp import UNet3dDp
from .ocenet import OCENET
from .gate_3d import GateUNet3D
from .BayesianModels.BayesianUnet_3d import BUNet3D
from .whatmodel import WhatModel
from methods.layer_ensembles import LayerEnsembles
from utils.le_utils import Task, Organ
###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_net(input_nc, output_nc, net_name, norm='batch', init_type='normal', init_gain=0.02,
               gpu_ids=[], output_shape=None):
    """
    Create a network based on the flags given in the options
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        net_inpaint (str) -- the inpainting architecture
        norm (str) -- the name of normalization layers used in the network: batch | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    The generator has been initialized by <init_net>.
    """
    net = None
    batch_norm = False
    if norm == 'batch':
        batch_norm = True
    print('net_name', net_name)
    if net_name == 'unet':
        net = UNet3Dv1(input_nc, output_nc)
    elif net_name == 'unet_c64':
        net = UNet3Dv1c64(input_nc, output_nc)
    elif net_name == 'gate_unet':
        net = GateUNet3D(input_nc, output_nc)
    elif net_name == 'ocenet':
        net = OCENET(input_nc, output_nc)
        # print(net)
    elif net_name == 'unet_le':
        architecture = UNet3D(input_nc, output_nc)
        all_layers = dict([*architecture.named_modules()])
        intermediate_layers = []
        for name, layer in all_layers.items():
            print('name', name)
            if '_2' in name and (len(name.split(".")) == 1):
                intermediate_layers.append(name)
        print('intermediate_layers', intermediate_layers)
        model = LayerEnsembles(architecture, intermediate_layers)

        # Dummy input to get the output shapes of the layers
        x = torch.zeros(output_shape)
        output = model(x)
        out_channels = []
        scale_factors = []
        for key, val in output.items():
            out_channels.append(val.shape[1])
            scale_factors.append(output_shape[-1] // val.shape[-1])
            # scale_factors.append(1)
        print('out_channesl', out_channels)
        print('scale_factors', out_channels)
        # Set the output heads with the number of channels of the output layers
        model.set_output_heads(in_channels=out_channels, scale_factors=scale_factors, task=Task.SR3D,
                               classes=45)
        net = model
    elif net_name == 'bayesian_unet':
        net = BUNet3D(input_nc, output_nc)
    elif net_name == 'unet_dp':
        net = UNet3dDp(input_nc, output_nc)
    elif net_name == 'aleatoric':
        net = WhatModel()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net_name)
    return init_net(net, init_type, init_gain, gpu_ids)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    x = torch.Tensor(1, 3, image_size, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))

    model = define_net(input_nc=3, output_nc=3, net_name='unet')

    out = model(x)
    print("out size: {}".format(out.size()))