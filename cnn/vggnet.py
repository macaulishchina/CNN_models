#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataloader import Cifar100Loader, Cifar10Loader
import classify
import framework

# gpu
CUDA_VISIBLE_DEVICES = '0, 1, 2, 3, 4, 5, 6, 7'
# 初始化训练参数
TAG = 'vgg16_cifar10_bs_1600'
TRAIN = True
TRAIN_WEIGHT_PATH = TAG + '.pkl'
TEST = True
SAVE_INTERVAL = 99999
TEST_WEIGHT_PATH = TAG + '.pkl'
BATCH_SIZE = 200
EPOCHS = '240,160'
LEARNING_RATES = '0.0001, 0.00005'
INPUT_SIZE = (128, 128)
NUM_CLASSES = 10
dataLoader = Cifar10Loader(input_size=INPUT_SIZE)
device, device_ids = framework.init_device(CUDA_VISIBLE_DEVICES)

# 训练模型
if TRAIN:
    vgg16 = framework.get_vgg16_model(weights_path=TRAIN_WEIGHT_PATH, num_classes=NUM_CLASSES)
    vgg16s = framework.parallelize_model(vgg16, device_ids)
    train_part = vgg16
    classify.train_schedule(TAG,
                            vgg16s,
                            also_test=True,
                            dataLoader=dataLoader,
                            epochs=EPOCHS,
                            lrs=LEARNING_RATES,
                            save_interval=SAVE_INTERVAL,
                            trainable=train_part,
                            DEVICE=device,
                            batch_size=BATCH_SIZE)

# 测试模型
if TEST:
    vgg16 = framework.get_vgg16_model(weights_path=TEST_WEIGHT_PATH, num_classes=NUM_CLASSES)
    vgg16s = framework.parallelize_model(vgg16, device_ids)
    classify.evaluate(vgg16s, DEVICE=device, batch_size=BATCH_SIZE, dataLoader=dataLoader)
