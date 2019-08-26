#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import models
from torch import nn
import framework

CUDA_VISIBLE_DEVICES = '1, 2'

DEVICE = torch.device("cuda:%s" % CUDA_VISIBLE_DEVICES if torch.cuda.is_available() else 'cpu')
print('Working on', 'gpu %s' % CUDA_VISIBLE_DEVICES if torch.cuda.is_available() else 'cpu')

features = models.vgg16(False).features
vgg16 = models.VGG(features, num_classes=10, init_weights=True)

framework.try_load_weights(vgg16, '[vgg16_cifar10:epoch5]2019-08-26 19:45:14.pkl')

net = nn.DataParallel(vgg16, device_ids=[int(id) for id in CUDA_VISIBLE_DEVICES.split(',')])
dataLoader = framework.XLoader('../data/cifar10', num_workers=1, input_size=(128, 128))
train_part = vgg16
framework.train_schedule('vgg16_cifar10',
                         net,
                         epochs=[20],
                         lrs=[0.0001],
                         save_interval=5,
                         trainable=train_part,
                         DEVICE=DEVICE,
                         batch_size=100)
framework.evaluate(net, DEVICE=DEVICE)
