import argparse


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # for general
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--dir_data', type=str, default='/media/lbs/HDD/VC_Dataset', help='path to data')
        g_data.add_argument('--dir_results', type=str, default='/home/lbs/VC/results', help='path to data')
        g_data.add_argument('--dir_checkpoints', type=str, default='/home/lbs/VC/checkpoints', help='path to data')

        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='2_6_1024_2dfrf_blur',
                           help='name of the experiment. It decides where to store samples and models')
        g_exp.add_argument('--debug', action='store_true', help='debug mode or not')

        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=4, help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', type=str, default=[4,5,6,7], help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        g_train.add_argument('--use_multi_gpus', default=True, help='load pretrained model')

        g_train.add_argument('--continue_train', default=False, help='load pretrained model')
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='resume epoch')

        g_train.add_argument('--num_threads', default=32, type=int, help='# sthreads for loading data')
        g_train.add_argument('--pin_memory', default=False, help='pin_memory')

        g_train.add_argument('--shuffle', default=True, help='shuffle training data')
        g_train.add_argument('--batch_size', type=int, default=32, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--schedule', type=int, nargs='+', default=[20, 60], help='Decrease learning rate at these epochs.')
        g_train.add_argument('--rf_schedule', type=int, nargs='+', default=[35, 36], help='Decrease learning rate at these epochs.')
        g_train.add_argument('--sigma_schedule', type=int, nargs='+', default=[10, 60], help='Decrease learning rate at these epochs.')

        g_train.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        g_train.add_argument('--num_epoch', type=int, default=200, help='num epoch to train')
        g_train.add_argument('--refine_epoch', type=int, default=35, help='heatmaps target weight')

        g_train.add_argument('--input_size', type=int, default=1024, help='input image size')
        g_train.add_argument('--num_joints', type=int, default=21, help='number of joints')
        g_train.add_argument('--sigma', type=float, default=4, help='heatmaps sigma')
        g_train.add_argument('--use_target_weight', default=False, help='heatmaps target weight')
        g_train.add_argument('--num_blocks', type=int, default=6, help='heatmaps target weight')
        g_train.add_argument('--num_channels', type=int, default=64, help='heatmaps target weight')
        g_train.add_argument('--aug_blur', default=True, help='heatmaps target weight')

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--test_folder_path', type=str, default=None, help='the folder of test image')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
