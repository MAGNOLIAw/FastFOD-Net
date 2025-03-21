from .base_options import BaseOptions

class TestOptions(BaseOptions):
    """
    This class includes test options with all shared options with train.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--view', type=str, default='ax', help='the view of the pseudo slices as input')
        parser.add_argument('--pad_to_size', type=int, default=-1, help='the size of the slice fed into the network')
        parser.add_argument('--print_stats', action='store_true', help='whether to print the inference metrics or not')
        parser.add_argument('--raw_output', action='store_true', help='whether the inference saves the raw network output')
        # rewrite devalue values
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')

        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--output_dir', type=str, default='./predictions', help='predictions are saved here')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
