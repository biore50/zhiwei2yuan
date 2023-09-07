#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import os
# torch
import torch
import torch.nn as nn
import torch.optim as optim


from .processor import Processor

import matplotlib.pyplot as plt

import pickle as pkl
from mmcv.cnn import constant_init, kaiming_init
from backbones.posefuser import Bottleneck3d, BasicBlock3d


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv3d):
        kaiming_init(m)
    elif isinstance(m, nn.BatchNorm3d):
        constant_init(m, 1)
    elif isinstance(m, Bottleneck3d):
        constant_init(m.conv3.bn, 0)
    elif isinstance(m, BasicBlock3d):
        constant_init(m.conv2.bn, 0)
    elif type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
    else:
        if type(m) == nn.Embedding:
            torch.nn.init.uniform_(m.weight)
        else:
            if type(m) == nn.GRU or type(m) == nn.LSTM:
                torch.nn.init.orthogonal_(m.weight_ih_l0)
                torch.nn.init.orthogonal_(m.weight_hh_l0)


class REC_Processor(Processor):

    def load_model(self):
        self.arg.model_args = dict(depth=50, pretrained=None, stage_blocks=None, pretrained2d=True, in_channels=3,
                                   k_channels=28, seq_len=16, num_classes=20, dropout_ratio=0.5, num_stages=3, base_channels=32)

        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))


        self.model.apply(initialize_weights)  # weights_init

        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
        return accuracy

    def evaluate_accuracy(self, data_iter, net):
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            acc_sum += (net(x).argmax(dim=1) == y).float().sum.item()
            n += y.shape[0]
        return acc_sum / n


    def train(self):
        train_l1_sum,train_l2_sum, train_acc_sum, n = 0.0, 0.0, 0.0, 0  # train_loss, num_correct,n样本总数

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']

        loss_value = []

        for image, lidar, label in loader:

            # get data
            image = image.float().to(self.dev)
            lidar = lidar.float().to(self.dev)
            label = label.long().to(self.dev)
            # print('image=',image.shape)

            # forward
            output = self.model(image,lidar)  # y_hat  output1,
            # output = output1 + output2
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0

            loss1 = self.loss(output, label).sum() + l1
            # loss2 = self.loss(output2, label).sum() + l2
            loss = self.loss(output, label).sum()

            # backward
            self.optimizer.zero_grad()
            loss1.backward()
            # loss2.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss1.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            train_l1_sum += loss1.item()  # train_loss
            # train_l2_sum += loss2.item()
            train_acc_sum += (output.argmax(dim=1) == label).sum().item()  # num_correct
            n += label.shape[0]


        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value1 = []
        loss_value2 = []
        result_frag = []
        label_frag = []
        acc_sum, n = 0.0, 0

        cmt = torch.zeros(len(self.attack_types), len(self.attack_types), dtype=torch.int64)

        for image, lidar, label in loader:
            # get data
            image = image.float().to(self.dev)
            lidar = lidar.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(image,lidar) #output1,
                # output3 = output1 + output2
                cmt = self.confusion_matrix(output, label, cmt)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0

                loss1 = self.loss(output, label)
                # loss2 = self.loss(output2, label)

                # valid_losses.append(loss.item())
                loss_value1.append(loss1.item())
                loss_value2.append(loss1.item())
                label_frag.append(label.data.cpu().numpy())
            acc_sum += (output.argmax(dim=1) == label).sum().item()
            n += label.shape[0]

        # self.val_writer.add_scalar('val_acc', acc_sum / n, self.meta_info['epoch'])
        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value1)
            self.show_epoch_info()
            accuracy = self.show_topk(k=1)
            accuracy5 = self.show_topk(k=5)
            if accuracy > self.best_acc:
                self.best_acc = accuracy


    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation   default=[1, 5]
        parser.add_argument('--show_topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        # parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
