import os
import datetime
import argparse
import shutil
import random
import numpy as np
import torch
from utils.tf_logger import TfLogger
import logging
import logging.config


class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN')

        parser.add_argument('--phase', default='train', type=str, help='train or test(default)')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')

        # dataset args
        parser.add_argument('--test_path', type=str, default='../bigRed_h5_gcn/')
        parser.add_argument('--in_channels', default=5, type=int, help='the channel size of input point cloud ')

        # train args
        parser.add_argument('--total_epochs', default=100, type=int, help='number of total epochs to run')
        parser.add_argument('--save_freq', default=1, type=int, help='save model per num of epochs')
        parser.add_argument('--iter', default=0, type=int, help='number of iteration to start')
        parser.add_argument('--lr_adjust_freq', default=20, type=int, help='decay lr after certain number of epoch')
        parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
        parser.add_argument('--lr_decay_rate', default=0.5, type=float, help='learning rate decay')
        parser.add_argument('--print_freq', default=100, type=int, help='print frequency of training (default: 100)')
        parser.add_argument('--postname', type=str, default='', help='postname of saved file')
        parser.add_argument('--multi_gpus', default=True, help='use multi-gpus')
        parser.add_argument('--gpuNum', default='0,1,2,3', help='use multi-gpus')

        parser.add_argument('--optim', type=str, default='adam', help='optimizer')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        # test args
        parser.add_argument('--visu', action='store_true', help='set --visu if need visualization in test phase')
        parser.add_argument('--no_clutter', action='store_true', help='no clutter? set --no_clutter if ture.')

        # model args
        parser.add_argument('--pretrained_model',default='gcn_model1/My_gcn-res-edge-4-128-200421_model_best.pth', type=str, help='path to pretrained model(default: none)')
        parser.add_argument('--block', default='res', type=str, help='graph backbone block type {res, dense}')
        parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
        # parser.add_argument('--conv', default='mr', type=str, help='graph conv layer {edge, mr}')

        parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm', default='instance', type=str, help='batch or instance normalization')
        parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
        parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')

        # dilated knn
        parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
        parser.add_argument('--stochastic', default=True, type=bool, help='stochastic for gcn, True or False')

        #need ajust
        parser.add_argument('--train_path', type=str, default='../bigRed_h5_gcn')
        parser.add_argument('--n_blocks', default=4, type=int, help='number of basic blocks')
        parser.add_argument('--n_filters', default=128, type=int, help='number of channels of deep features')
        parser.add_argument('--kernel_size', default=3, type=int, help='neighbor num (default:16)')
        parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size (default:16)')
        parser.add_argument('--num_channel', default=4, type=int, help='mini-batch size (default:16)')





        args = parser.parse_args()
        args.in_channels = args.num_channel
        dir_path = os.path.dirname(os.path.abspath(__file__))
        args.task = os.path.basename(dir_path)
        args.post = '-'.join([args.task, args.block, args.conv, str(args.n_blocks), str(args.n_filters)])
        if args.postname:
            args.post += '-' + args.postname
        args.post += '-' + datetime.datetime.now().strftime("%y%m%d")

        if args.pretrained_model:
            if args.pretrained_model[0] != '/':
                if args.pretrained_model[0:2] == 'ex':
                    args.pretrained_model = os.path.join(os.path.dirname(os.path.dirname(dir_path)),
                                                         args.pretrained_model)
                else:
                    args.pretrained_model = os.path.join(dir_path, args.pretrained_model)
                args.pretrained_model = os.path.join(dir_path, args.pretrained_model)

        args.save_path = os.path.join(dir_path, 'checkpoints/ckpts' + '-' + args.post)
        args.logdir = os.path.join(dir_path, 'logs/' + args.post)

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args

    def initialize(self):
        if self.args.phase == 'train':
            # logger
            self.args.logger = TfLogger(self.args.logdir)
            # loss
            self.args.epoch = -1
            self.make_dir()

        self.set_seed(self.args.seed)
        self.logging_init()
        self.print_args()
        return self.args

    def print_args(self):
        self.args.printer.info("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            self.args.printer.info("{}:{}".format(arg, content))
        self.args.printer.info("==========     CONFIG END    =============")
        self.args.printer.info("\n")
        self.args.printer.info('===> Phase is {}.'.format(self.args.phase))

    def logging_init(self):
        if not os.path.exists(self.args.logdir):
            os.makedirs(self.args.logdir)
        ERROR_FORMAT = "%(message)s"
        DEBUG_FORMAT = "%(message)s"
        LOG_CONFIG = {'version': 1,
                      'formatters': {'error': {'format': ERROR_FORMAT},
                                     'debug': {'format': DEBUG_FORMAT}},
                      'handlers': {'console': {'class': 'logging.StreamHandler',
                                               'formatter': 'debug',
                                               'level': logging.INFO},
                                   'file': {'class': 'logging.FileHandler',
                                            'filename': os.path.join(self.args.logdir, self.args.post + '.log'),
                                            'formatter': 'debug',
                                            'level': logging.INFO}},
                      'root': {'handlers': ('console', 'file'), 'level': 'INFO'}
                      }
        logging.config.dictConfig(LOG_CONFIG)
        self.args.printer = logging.getLogger(__name__)

    def make_dir(self):
        # check for folders existence
        shutil.rmtree(self.args.logdir)
        os.makedirs(self.args.logdir)

        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        if not os.path.exists(self.args.train_path):
            os.makedirs(self.args.train_path)

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
