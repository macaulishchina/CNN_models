#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataloader import Cifar100Loader
import classify
import framework

# gpu
CUDA_VISIBLE_DEVICES = '0, 1, 2, 3, 4, 5, 6, 7'
# 初始化训练参数
TAG = 'vgg16_cifar100_bs_1600'
BATCH_SIZE = 1600
EPOCHS = '200, 200, 200, 200'
LEARNING_RATES = '0.0001, 0.00005, 0.00001, 0.000005'
INPUT_SIZE = (128, 128)
SAVE_INTERVAL = 100
NUM_CLASSES = 100

# 训练模型
device, device_ids = framework.init_device(CUDA_VISIBLE_DEVICES)
vgg16 = framework.get_vgg16_model(weights_path='', num_classes=NUM_CLASSES)
vgg16s = framework.parallelize_model(vgg16, device_ids)
trainloader = Cifar100Loader(input_size=INPUT_SIZE)
train_part = vgg16
classify.train_schedule(TAG,
                        vgg16s,
                        dataLoader=trainloader,
                        epochs=EPOCHS,
                        lrs=LEARNING_RATES,
                        save_interval=10,
                        trainable=train_part,
                        DEVICE=device,
                        batch_size=BATCH_SIZE)

# 测试模型
vgg16 = framework.get_vgg16_model(weights_path=TAG + '.pkl', num_classes=NUM_CLASSES)
testloader = framework.Cifar100Loader(input_size=INPUT_SIZE)
classify.evaluate(vgg16, DEVICE=device, batch_size=BATCH_SIZE, dataLoader=testloader)
