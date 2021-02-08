from collections import OrderedDict

import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from operator import itemgetter
import time
import tensorly as tl
import tensorly
from itertools import chain
from decomposition import torch_cp_decomp
from decompositionall import torch_cp_decomp_all


class FastAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(FastAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        encode_features = self.encoder(inputs)
        # output = encode_features
        output = self.decoder(encode_features)
        return output


# in out w h padding stride
encoder_size = [512, 512, 512, 256, 256, 256, 128, 64, 32]
decoder_size = [32, 64, 128, 256, 256, 256, 512, 512, 512]


# decoder_size = encoder_size[::-1]


# 定义网络结构
def make_layers(encoder_size, decoder_size, batch_norm=True):
    encoder = OrderedDict()
    decoder = OrderedDict()
    # 输入的batchsize
    input_channels = 288
    output_channels = input_channels
    encoder_kernel_size = 2
    decoder_kernel_size = 2
    padding_size = 0
    stride_size = 1
    order = 1
    order2 = 1
    for i, v in enumerate(encoder_size):

        if batch_norm:
            encoder['conv' + str(order)] = nn.Conv2d(input_channels, v, encoder_kernel_size,
                                                     stride=stride_size, padding=padding_size, bias=True
                                                     )
            encoder['bt' + str(order)] = nn.BatchNorm2d(v)
            encoder['Sigmoid' + str(order)] = nn.Sigmoid()
            input_channels = v
            order += 1
        else:
            encoder['conv' + str(order)] = nn.Conv2d(input_channels, v, encoder_kernel_size,
                                                     stride=stride_size, padding=padding_size, bias=True
                                                     )
            encoder['Sigmoid' + str(order)] = nn.Sigmoid()
            input_channels = v
            order += 1

    for i, v in enumerate(decoder_size):
        if i < len(decoder_size)-1:
            if batch_norm:
                decoder['conv' + str(order2)] = nn.ConvTranspose2d(input_channels, v * 2, decoder_kernel_size,
                                                                   stride=stride_size, padding=padding_size, bias=True
                                                                   )
                decoder['bt' + str(order2)] = nn.BatchNorm2d(v * 2)
                decoder['Sigmoid' + str(order2)] = nn.Sigmoid()
                input_channels = v * 2
                order2 += 1
            else:
                decoder['conv' + str(order2)] = nn.ConvTranspose2d(input_channels, v * 2, decoder_kernel_size,
                                                                   stride=stride_size, padding=padding_size, bias=True
                                                                   )
                decoder['Sigmoid' + str(order2)] = nn.Sigmoid()
                input_channels = v * 2
                order2 += 1
        else:
            if batch_norm:
                decoder['conv' + str(order2)] = nn.ConvTranspose2d(input_channels, output_channels, decoder_kernel_size,
                                                                   stride=stride_size, padding=padding_size, bias=True
                                                                   )
                decoder['bt' + str(order2)] = nn.BatchNorm2d(output_channels)
                decoder['Sigmoid' + str(order2)] = nn.Sigmoid()
                input_channels = output_channels
                order2 += 1
            else:
                decoder['conv' + str(order2)] = nn.ConvTranspose2d(input_channels, output_channels, decoder_kernel_size,
                                                                   stride=stride_size, padding=padding_size, bias=True
                                                                   )
                decoder['Sigmoid' + str(order2)] = nn.Sigmoid()
                input_channels = output_channels
                order2 += 1

    return nn.Sequential(encoder), nn.Sequential(decoder)


def model_set(**kwargs):
    encoder, decoder = make_layers(encoder_size, decoder_size, batch_norm=True)
    model = FastAutoEncoder(encoder, decoder, **kwargs)
    return model


# 定义训练器
class Trainer:
    def __init__(self, train_data, model, optimizer, epoch=1000):
        # 传参
        self.train_data = train_data
        self.optimizer = optimizer
        self.model = model
        self.epoch = epoch

        # 定义
        # L1Loss MSELoss
        self.criterion = torch.nn.SmoothL1Loss()
        self.model.train()

    def test(self):
        self.model.cpu()
        self.model.eval()
        t0 = time.time()
        output = self.model(self.train_data)
        t1 = time.time()
        return output
        print("Average prediction time", float(t1 - t0))

        self.model.train()

    def train(self):
        total_time = 0
        t0 = time.time()
        for i in range(self.epoch):
            loss_train = self.train_batch()
            # print('Epoch :', i, ';train_loss:%.4f' % loss_train.data)
        t1 = time.time()
        total_time = total_time + (t1 - t0)
        print("Training time:", float(total_time))
        # print("Finished fine tuning.")

    def train_batch(self):
        input_data = self.train_data.cpu()
        self.model.zero_grad()
        output = self.model(input_data)
        loss_train = self.criterion(output, input_data)
        loss_train.backward()
        self.optimizer.step()
        return loss_train


def low_rank(normal_data, learn_ratio, cp_rank, if_use=False):
    tl.set_backend('pytorch')
    low_rank_net = model_set()
    low_rank_net.cpu()
    input_data = torch.Tensor(normal_data)
    input_data = torch.abs(input_data)
    input_data = input_data.cpu()
    # 优化函数还可以是torch.optim.SGD
    optimizer = torch.optim.Adam(low_rank_net.parameters(), lr=learn_ratio)
    if not if_use:
        # 先训练一个模型
        first_epoch = 20000

        trainer = Trainer(input_data, low_rank_net, optimizer, first_epoch)
        t_basic0 = time.time()
        trainer.train()
        t_basic1 = time.time()
        print('basic:', t_basic1-t_basic0)
        torch.save(low_rank_net, "basic_model_cpu")
        print("Finished basic model.")

        # 将上述模型进行分解
        low_rank_net.eval()
        low_rank_net.cpu()
        length_encoder = len(low_rank_net.encoder._modules.keys())
        length_decoder = len(low_rank_net.decoder._modules.keys())

        # # 对于encoder
        for i, key in enumerate(low_rank_net.encoder._modules.keys()):
            tran = False
            if i >= length_encoder - 1:
                break
            if isinstance(low_rank_net.encoder._modules[key], nn.modules.conv.Conv2d):
                conv_layer_encoder = low_rank_net.encoder._modules[key]
                # print(conv_layer_encoder)
                # decomposed_encoder = torch_cp_decomp(conv_layer_encoder, cp_rank, tran)
                decomposed_encoder = torch_cp_decomp_all(conv_layer_encoder, cp_rank, tran)
                low_rank_net.encoder._modules[key] = decomposed_encoder

        # 对于decoder
        for i, key in enumerate(low_rank_net.decoder._modules.keys()):
            tran = True
            if i >= length_decoder - 1:
                break
            if isinstance(low_rank_net.decoder._modules[key], nn.modules.conv.ConvTranspose2d):
                conv_layer_decoder = low_rank_net.decoder._modules[key]
                # print(conv_layer_decoder)
                # decomposed_decoder = torch_cp_decomp(conv_layer_decoder, cp_rank, tran)
                decomposed_decoder = torch_cp_decomp_all(conv_layer_decoder, cp_rank, tran)
                low_rank_net.decoder._modules[key] = decomposed_decoder

        torch.save(low_rank_net, 'decomposed_model_cpu')
        print('Finish decomposed')

        # fine_tune模型
        fine_tune_epoch = 20000
        # model_decompose = torch.nn.DataParallel(low_rank_net)
        model_decompose = low_rank_net
        for param in model_decompose.parameters():
            param.requires_grad = True

        model_decompose.cpu()
        optimizer_decompose = optim.Adam(model_decompose.parameters(), lr=0.001)

        trainer = Trainer(input_data, model_decompose, optimizer_decompose, fine_tune_epoch)
        t_dec0 = time.time()
        trainer.train()
        t_dec1 = time.time()
        print('decomposition:', t_dec1-t_dec0)
        torch.save(model_decompose, 'model_decompose_fine_tune_cpu')
        print("Finished fine tuning.")
        output = model_decompose(input_data)
        return output
    else:
        model = torch.load("basic_model_cpu").cuda()
        t0 = time.time()
        output = model(input_data)
        t1 = time.time()
        print("CAE process time", float(t1 - t0))
        return output


if __name__ == '__main__':
    print('1')

