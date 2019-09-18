# -*- coding: utf-8 -*-
import os
import torch
import json
import numpy as np
from torch import nn
from torchvision import models
from matplotlib import pyplot as plt
from args import args
import dataloader


def get_dataloader_by_name(name):

    num_workers = args.load_workers
    input_size = (args.input_size, args.input_size)
    enhancement = args.enhancement
    tencrop = args.tencrop
    root = '/dev/shm' if args.share_memory else ('../data/%s' % name)
    if name == 'cifar10':
        return dataloader.Cifar10Loader(args.download, root=root, num_workers=num_workers, input_size=input_size, enhancement=enhancement)
    if name == 'cifar100':
        return dataloader.Cifar100Loader(args.download, root=root, num_workers=num_workers, input_size=input_size, enhancement=enhancement)
    if name == 'caltech256':
        return dataloader.Caltech256Loader(root=root, num_workers=num_workers, input_size=input_size, enhancement=enhancement, tencrop=tencrop)

s = models.detection.fasterrcnn_resnet50_fpn(False, pretrained_backbone=False)
print(s)


def get_model_by_name(name, num_classes):
    weights_name = args.weights_name
    pretrained = args.pretrained
    if name == 'vgg16':
        return get_vgg16_model(weights_name, num_classes, pretrained)
    if name == 'googLenet':
        return get_googLenet_model(weights_name, num_classes)
    if name == 'resnet50':
        return get_resnet_50_model(weights_name, num_classes)


def init_device(gpu='0'):
    DEVICE = torch.device("cuda:%s" % gpu if torch.cuda.is_available() else 'cpu')
    print('Working on', 'gpu %s' % gpu if torch.cuda.is_available() else 'cpu')
    return DEVICE, [int(id) for id in gpu.split(',')]


def try_load_weights(model, file, weights_dir='./weights'):
    weights_path = '%s/%s' % (weights_dir, file)
    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print('Successfully load weights from path: `' + weights_path + '`')
    else:
        print('\nWeights file not found in path: `' + weights_path + '`.Use default weights instead.\n')
    return model


def get_vgg16_model(weights_path=None, num_classes=1000, pretrained=False):
    vgg16 = models.vgg16(pretrained, num_classes=num_classes, init_weights=True)
    try_load_weights(vgg16, weights_path)
    return vgg16


def get_googLenet_model(weights_path=None, num_classes=1000):
    googLenet = models.GoogLeNet(num_classes=num_classes, init_weights=True, aux_logits=False)
    try_load_weights(googLenet, weights_path)
    return googLenet


def get_resnet_50_model(weights_path=None, num_classes=1000):
    resnet50 = models.resnet50(False, num_classes=num_classes)
    try_load_weights(resnet50, weights_path)
    return resnet50


def parallelize_model(model, device_ids):
    return nn.DataParallel(model, device_ids)


def visualize_feedback(feedback=None,
                       feedback_file=None,
                       feedback_file_list=[],
                       tag='feedback',
                       save_dir='./weights',
                       also_test=False
                       ):
    """可视化训练结果
    """
    assert feedback is not None and isinstance(feedback, dict) or feedback_file is not None or feedback_file_list != []
    if feedback_file is not None:
        with open(save_dir + '/' + feedback_file, 'r') as file:
            feedback = json.load(file)
    feedback = {} if feedback is None else feedback
    if feedback_file_list != []:
        last_epoch = len(feedback.keys())
        for file in feedback_file_list:
            with open(save_dir + '/' + file, 'r') as file:
                sub_feedback = json.load(file)
            for k, v in sub_feedback.items():
                feedback[last_epoch + 1] = v
                last_epoch += 1
    with open(save_dir + '/' + tag + '.json', 'w') as file:
        json.dump(feedback, file)
    opoch_x = list(feedback.keys())
    train_accuracy = []
    test_accuracy = []
    loss = []
    lr = []
    batch_size = []
    last_lr = 0
    group_indexes = []
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('epoch', fontsize=20)
    ax1.set_ylabel('loss', fontsize=20)

    ax1.set_title(tag, fontsize=25)
    for epoch in opoch_x:
        fd = feedback[epoch]
        if fd[2] != last_lr:
            group_indexes.append([0, 1] if group_indexes == [] else [group_indexes[-1][1] - 1, group_indexes[-1][1] + 1])
            last_lr = fd[2]
        else:
            group_indexes[-1][1] += 1
        train_accuracy.append(fd[0])
        if also_test:
            test_accuracy.append(fd[4])
        loss.append(fd[1])
        lr.append(fd[2])
        batch_size.append(fd[3])
    for group_index in group_indexes:
        x = np.array(opoch_x, dtype=np.int32)[group_index[0]:group_index[1]]
        loss_y = np.array(loss)[group_index[0]:group_index[1]]
        ax1.plot(x, loss_y, label='loss(lr:%s)' % lr[group_index[-1] - 1], linestyle='--', linewidth=2)
    x = np.array(opoch_x, dtype=np.int32)
    train_accuracy_y = np.array(train_accuracy) * 100
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy(%)', fontsize=20)
    ax2.plot(x, train_accuracy_y, label="train accuracy", color='orange', linewidth=2)
    if also_test:
        test_accuracy_y = np.array(test_accuracy) * 100
        ax2.plot(x, test_accuracy_y, label="test accuracy", color='green', linewidth=2)
    fig.legend(bbox_to_anchor=(0.90, 0.35))
    save_path = '%s/%s.jpg' % (save_dir, tag)
    plt.savefig(save_path)
    print('Visualize result saved to `%s`' % save_path)
