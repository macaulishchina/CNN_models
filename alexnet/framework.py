# -*- coding: utf-8 -*-
import os
import time
import math
import json
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from matplotlib import pyplot as plt

import progressbar as pb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XLoader():

    def __init__(self, root='../data/cifar10', num_workers=0):
        self.root = root
        self.num_workers = num_workers
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.Resize((227, 227)),  # AlexNet的输入尺寸
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # R,G,B每层的归一化用到的均值和方差
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((227, 227)),  # AlexNet的输入尺寸
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # R,G,B每层的归一化用到的均值和方差
        ])
        self.trainset = CIFAR10(self.root, train=True, transform=train_transforms, download=True)
        self.testset = CIFAR10(self.root, train=False, transform=test_transforms, download=True)

    def get_train_loader(self, batch_size=128, shuffle=True):
        return DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_test_loader(self, batch_size=128, shuffle=False):
        return DataLoader(self.testset, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)


def train(model,
          dataLoader,
          lr=0.001,
          optimizer=None,
          loss_F=None,
          epochs=10,
          save_interval=10,
          save_dir='./weights',
          batch_size=128,
          last_epoch=0,
          shuffle=True,
          print_feedback=True,
          tag=None,
          DEVICE=DEVICE,
          trainable=None,
          ):
    model.to(DEVICE)
    model.train()
    loss_F = nn.CrossEntropyLoss() if loss_F is None else loss_F
    trainable = model if trainable is None else trainable
    optimizer = optim.Adam(trainable.parameters(), lr=lr) if optimizer is None else optimizer
    feedback = {}
    last_epoch_accuracy = 0
    last_epoch_loss = 2 ** 32
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batch_count = 0
        train_loader = dataLoader.get_train_loader(batch_size, shuffle)
        epoch = epoch + last_epoch + 1
        print('Training epoch %s/%s, sum=%s, batch_size=%s, lr=%s ...' % (epoch, epochs + last_epoch, len(train_loader.dataset), batch_size, lr))
        weights = [pb.Percentage(), '(', pb.SimpleProgress(), ')', pb.Bar('>'), '[accuracy=%0.3f%%, loss=%.5s ]' % (0, 0), pb.ETA()]
        with pb.ProgressBar(max_value=math.ceil(len(train_loader.dataset) / batch_size), widgets=weights) as bar:
            bar.update(0)
            last_batch_accuracy = 0
            last_batch_loss = 0
            for inputs, labels in train_loader:
                batch_count += 1
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                batch_loss = loss_F(outputs, labels)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                batch_accuracy = torch.max(outputs.data, 1)[1].eq(labels.data).cpu().sum().item() / labels.size(0)
                epoch_accuracy += batch_accuracy
                mark_accuracy = 'ꜛ' if batch_accuracy > last_batch_accuracy else 'ꜜ'
                mark_loss = 'ꜛ' if batch_loss.item() > last_batch_loss else 'ꜜ'
                last_batch_accuracy = batch_accuracy
                last_batch_loss = batch_loss.item()
                weights[5] = '[accuracy=%0.3f%%%s, loss=%.5f%s]' % (batch_accuracy * 100, mark_accuracy, batch_loss.item(), mark_loss)
                bar.update(batch_count)
        loss = epoch_loss / batch_count
        accuracy = epoch_accuracy / batch_count
        mark_accuracy = 'ꜛ' if accuracy > last_epoch_accuracy else 'ꜜ'
        mark_loss = 'ꜛ' if loss > last_epoch_loss else 'ꜜ'
        last_epoch_accuracy = accuracy
        last_epoch_loss = loss
        feedback[epoch] = [accuracy, loss, lr, batch_size]
        if print_feedback:
            print("Epoch train report: accuracy = %0.3f%%%s, loss = %.5f%s.\n" % (accuracy * 100, mark_accuracy, loss, mark_loss))
        if epoch % save_interval == 0:
            file_name = '[%sepoch%s]%s.pkl' % ('' if tag is None else '%s:' % tag, epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            save_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(save_dict, save_dir + '/' + file_name)
            print('Weights file saved to `%s`\n' % (save_dir + '/' + file_name))
    return model, feedback


def test(model,
         dataloader,
         dataset_type='',
         print_feedback=True,
         batch_size=128,
         DEVICE=DEVICE
         ):
    model.to(DEVICE)
    model.eval()
    epoch_accuracy = 0
    batch_count = 0
    print('Testing [%s], sum=%s, batch_size=%s ...' % (dataset_type, len(dataloader.dataset), batch_size))
    weights = [pb.Percentage(), '(', pb.SimpleProgress(), ')', pb.Bar('>'), '[accuracy=%0.3f%%]' % (0), pb.ETA()]
    with pb.ProgressBar(max_value=math.ceil(len(dataloader.dataset) / batch_size), widgets=weights) as bar:
        for inputs, labels in dataloader:
            batch_count += 1
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            batch_accuracy = torch.max(outputs.data, 1)[1].eq(labels.data).cpu().sum().item() / labels.size(0)
            epoch_accuracy += batch_accuracy
            weights[5] = '[accuracy=%0.3f%%]' % (batch_accuracy * 100)
            bar.update(batch_count)
        accuracy = epoch_accuracy / batch_count
    if print_feedback:
        print("Test report: accuracy = %0.3f%%.\n" % (accuracy * 100))
    return accuracy


def init_model(model):
    print('初始化所有参数为0.5')
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.constant_(module.weight.data, 0.5)
            nn.init.constant_(module.bias.data, 0)
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight.data, 0.5)
            nn.init.constant_(module.bias.data, 0.5)


def train_schedule(tag,
                   model,
                   loss_F=None,
                   optimizer=None,
                   epochs=[10],
                   lrs=[0.001],
                   batch_size=512,
                   trainable=None,
                   DEVICE=DEVICE,
                   dataLoader=None,
                   save_interval=10,
                   save_feedback=True,
                   print_feedback=True,
                   weights_dir='./weights'
                   ):
    assert len(epochs) == len(lrs)
    dataLoader = XLoader(num_workers=2) if dataLoader is None else dataLoader
    last_epoch = 0
    feedback = {}
    for idx, lr in enumerate(lrs):
        model, fb = train(model, dataLoader,
                          tag=tag, batch_size=batch_size, DEVICE=DEVICE,
                          epochs=epochs[idx], lr=lr, trainable=trainable,
                          save_interval=save_interval, optimizer=optimizer,
                          save_dir=weights_dir, last_epoch=last_epoch)
        last_epoch += epochs[idx]
        feedback.update(fb)
    weights_file = '%s/%s.pkl' % (weights_dir, tag)
    feedback_file = '%s/%s.json' % (weights_dir, tag)
    save_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(save_dict, weights_file)
    with open(feedback_file, 'w') as file:
        json.dump(feedback, file)
    visualize_feedback(feedback=feedback, save_dir=weights_dir, tag=tag)
    return model, feedback


def try_load_weights(model, file, weights_dir='./weights'):
    weights_path = '%s/%s' % (weights_dir, file)
    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print('Successfully load weights from path: `' + weights_path + '`')
    else:
        print('\nWeights file not found in path: `' + weights_path + '`.Use default weights instead.\n')
    return model


def evaluate(model, dataloader=None, batch_size=512, DEVICE=DEVICE):
    dataloader = XLoader() if dataloader is None else dataloader
    trainset = dataloader.get_train_loader(batch_size=batch_size, shuffle=False)
    testset = dataloader.get_test_loader(batch_size=batch_size, shuffle=False)
    train_accuracy = test(model, trainset, 'trainset', batch_size=batch_size, DEVICE=DEVICE)
    test_accuracy = test(model, testset, 'testset', batch_size=batch_size, DEVICE=DEVICE)
    return train_accuracy, test_accuracy


def visualize_feedback(feedback=None, feedback_file=None, tag='feedback', save_dir='./weights'):
    """可视化训练结果
    """
    assert feedback is not None and isinstance(feedback, dict) or feedback_file is not None
    if feedback_file is not None:
        with open(feedback_file, 'r') as file:
            feedback = json.load(file)
    opoch_x = list(feedback.keys())
    accuracy = []
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
            group_indexes.append([0, 1] if group_indexes == [] else [group_indexes[-1][1] + 1, group_indexes[-1][1] + 1])
            last_lr = fd[2]
        else:
            group_indexes[-1][1] += 1
        accuracy.append(fd[0])
        loss.append(fd[1])
        lr.append(fd[2])
        batch_size.append(fd[3])
    for group_index in group_indexes:
        x = np.array(opoch_x, dtype=np.int32)[group_index[0]:group_index[1]]
        loss_y = np.array(loss)[group_index[0]:group_index[1]]
        ax1.plot(x, loss_y, label='loss(lr:%s)' % lr[group_index[0]], linewidth=2)
    x = np.array(opoch_x, dtype=np.int32)
    accuracy_y = np.array(accuracy) * 100
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy(%)', fontsize=20)
    ax2.plot(x, accuracy_y, label="accuracy", color='green', linestyle='--', linewidth=2)
    fig.legend(bbox_to_anchor=(0.85, 0.65))
    plt.savefig('%s/%s.jpg' % (save_dir, tag))
