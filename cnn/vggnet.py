#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: 胡怡冬 <macaulish>
# @Date:   2019-08-30T14:00:07+08:00
# @Email:  huyidongchina@Gmail.com
# @Last modified by:   macaulish
# @Last modified time: 2019-08-30T14:00:27+08:00


from dataloader import ILSVRC2012Loader
import classify
import framework

# gpu
CUDA_VISIBLE_DEVICES = '0, 1, 2, 3, 4, 5, 6, 7'
# 初始化训练参数
MODEL_NAME = 'vgg16'
DATASET_NAME = 'ILSVRC2012'
BATCH_SIZE = 400
EPOCHS = '5'
LEARNING_RATES = '0.0001'
INPUT_SIZE = (224, 224)
NUM_CLASSES = 1000
TAG = '%s_on_%s_bs_%s' % (MODEL_NAME, DATASET_NAME, BATCH_SIZE)
TRAIN = True
TEST = True
SAVE_INTERVAL = 1
TRAIN_WEIGHT_PATH = TAG + '.pkl'
TEST_WEIGHT_PATH = TAG + '.pkl'
PRETRAINED = True
device, device_ids = framework.init_device(CUDA_VISIBLE_DEVICES)

dataLoader = ILSVRC2012Loader(input_size=INPUT_SIZE, enhancement=True, tencrop=False)

# 训练模型
if TRAIN:
    vgg16 = framework.get_vgg16_model(weights_path=TRAIN_WEIGHT_PATH, num_classes=NUM_CLASSES, pretrained=True)
    vgg16s = framework.parallelize_model(vgg16, device_ids)
    train_part = vgg16.classifier
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
