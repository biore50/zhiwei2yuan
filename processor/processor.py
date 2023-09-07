#!/usr/bin/env python
# pylint: disable=W0201
import sys, os
import argparse


# torch
import torch


# torchlight
# import torchlight
# from torchlight.torchlight import str2bool
# from torchlight.torchlight import DictAction
# from torchlight import import_class

from .io import IO
from torch.utils.tensorboard import SummaryWriter
from feeder.feeder import Feeder
# from tensorboardX import SummaryWriter

class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()
        self.arg.use_gpu = True
        self.arg.print_log = True
        self.arg.save_log = True
        self.arg.save_result = False
        self.arg.pavi_log = False
        self.rgb_path= '/rgb/'
        self.ske_path='/ske/'
        self.label_path = '/label.txt'
        self.arg.train_feeder_args = dict(dataset='gcf', split='train', clip_len= 4, preprocess=False, rgb_path=self.rgb_path,
                                          ske_path=self.ske_path,label_path=self.label_path)
        self.arg.test_feeder_args = dict(dataset='gcf', split='test', clip_len=4, preprocess=False,
                                          rgb_path=self.rgb_path,
                                          ske_path=self.ske_path, label_path=self.label_path)
        self.arg.model_args = dict(depth=50, pretrained=None,stage_blocks= None,pretrained2d= True,in_channels= 3, k_channels= 28,
                                   seq_len=16,num_classes=20,dropout_ratio= 0.5, num_stages = 3, base_channels= 32)


    def init_environment(self):
        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass

    def load_data(self):
        self.rgb_path = '/rgb/'
        self.ske_path = '/ske/'
        self.label_path = '/label.txt'
        self.arg.train_feeder_args = dict(dataset='gcf', split='train', clip_len=16, preprocess=False,
                                          rgb_path=self.rgb_path,
                                          ske_path=self.ske_path, label_path=self.label_path)
        self.arg.test_feeder_args = dict(dataset='gcf', split='test', clip_len=16, preprocess=False,
                                         rgb_path=self.rgb_path,
                                         ske_path=self.ske_path, label_path=self.label_path)


        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True, pin_memory=True
            )
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker , pin_memory=True
            )

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # training phase
        if self.arg.phase == 'train':
            # self.early_stopping = EarlyStopping(patience=20, verbose=True)
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.test()
                    self.io.print_log('Done.')



        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test()
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default='', help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')

        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        # parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100,
                            help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10,
                            help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5,
                            help='the interval for evaluating models (#iteration)')


        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')

        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')
        parser.add_argument('--class_num', type=int, default=256, help='training class num')


        # model
        parser.add_argument('--model', default=None, help='the model will be used')

        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            help='the name of weights which will be ignored in the initialization')


        return parser
