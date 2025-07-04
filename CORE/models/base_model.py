from utils.logger import setup_logger
logger = setup_logger()

import warnings
import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks

class BaseModel(ABC):
    """
    This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        torch.backends.cudnn.benchmark = True
        
        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        # Verbosity-based logging level
        # if getattr(opt, 'verbose', False):
        #     logger.setLevel(logging.DEBUG)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # if self.isTrain:
        if self.opt.phase == "train":
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        # if not self.isTrain or opt.continue_train:
        if self.opt.phase != "train" or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

    def test(self):
        """
        Forward function used in test time.

        BatchNorm behaves unexpected if using test(), so we use eval() and set to 'torch.no_grad'
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

        with torch.no_grad():
            output = self.forward()
            return output

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        logger.info('Learning rate %.7f → %.7f' % (old_lr, lr))        
        # print('learning rate %.7f -> %.7f' % (old_lr, lr))
        

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                logger.info(f'Saving model to: {save_path}')
                # print(f'Saving model to:{save_path}')
                net = getattr(self, 'net_' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch, load_path=None, weights_only=True):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
            load_path (str)     -- optional full path to the model file
            weights_only (bool) -- whether to load only state_dict (recommended for security)            
        """
        for name in self.model_names:
            if not isinstance(name, str):
                warnings.warn(f"Skipping non-string model name in model_names: {name} (type: {type(name)})", UserWarning)
                continue

            if load_path is None:
                load_filename = f'{epoch}_net_{name}.pth'
                path = os.path.join(self.save_dir, load_filename)
            else:
                path = load_path    

            net = getattr(self, 'net_' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            logger.info(f'[Loading] Model from: {path} ({type(net).__name__})')
            # print(f'Loading the model from {path} ({type(net).__name__})')

            # Load state dict securely
            # state_dict = torch.load(path, map_location=self.device, weights_only=weights_only)
            try:
                state_dict = torch.load(path, map_location=self.device, weights_only=weights_only)
            except TypeError:
                # Fallback for older PyTorch versions that do not support `weights_only`
                state_dict = torch.load(path, map_location=self.device)

            # Clean legacy instance norm incompatibilities
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            # patch InstanceNorm checkpoints prior to 0.4
            for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

            net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        # print('---------- Networks initialized -------------')
        # logger.info("═" * 25 + " Model Initialization " + "═" * 25)
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    logger.info(str(net))
                    # print(net)
                logger.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                # print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        # logger.info("═" * 70)
        logger.info("-" * 55)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad