#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision import models
from torch import nn
import framework

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Working on', 'gpu' if torch.cuda.is_available() else 'cpu')


def get_modified_pretrained_model(num_classes=10):
    alexnet = models.alexnet(True)
    pretrained_dict = alexnet.state_dict()
    w6 = pretrained_dict.pop('classifier.6.weight')
    b6 = pretrained_dict.pop('classifier.6.bias')
    classifier_6_weight = torch.empty(num_classes, w6.size()[1], dtype=w6.dtype)
    classifier_6_bias = torch.empty(num_classes, dtype=b6.dtype)
    nn.init.normal_(classifier_6_weight, 0, 0.05)
    nn.init.constant_(classifier_6_bias, 0)
    pretrained_dict['classifier.6.weight'] = classifier_6_weight
    pretrained_dict['classifier.6.bias'] = classifier_6_bias

    alexnet_10 = models.alexnet(True)
    alexnet_10.classifier[6] = nn.Linear(4096, 10)
    alexnet_10.load_state_dict(pretrained_dict)
    return alexnet_10


alexnet = get_modified_pretrained_model()

# framework.try_load_weights(alexnet, 'big_batch.pkl')

mul_alexnet = nn.DataParallel(alexnet, device_ids=[0, 1, 2, 3])
train_part = alexnet.classifier
framework.train_schedule('test_visualize',
                         mul_alexnet,
                         epochs=[5, 5],
                         lrs=[0.001, 0.0005],
                         save_interval=1,
                         trainable=train_part,
                         DEVICE=DEVICE,
                         batch_size=4000)
framework.evaluate(alexnet, DEVICE=DEVICE)
