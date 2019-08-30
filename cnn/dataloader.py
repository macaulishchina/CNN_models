#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100


class Loader():

    def __init__(self, root, num_workers, input_size, download, enhancement):
        self.root = root
        self.num_workers = num_workers
        self.input_size = input_size
        self.download = download
        self.enhancement = enhancement
        self.train_transforms = self.__generate_train_transforms__()
        self.test_transforms = self.__generate_test_transforms__()
        self.trainset = None
        self.testset = None

    def __generate_train_transforms__(self) -> transforms:
        pass

    def __generate_test_transforms__(self) -> transforms:
        pass

    def __generate_trainset__(self):
        pass

    def __generate_testset__(self):
        pass

    def get_train_loader(self, batch_size=1, shuffle=True):
        self.trainset = self.__generate_trainset__() if self.trainset is None else self.trainset
        assert self.trainset is not None, '请实现`_generate_trainset`方法，返回训练集'
        return DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_test_loader(self, batch_size=1, shuffle=False):
        self.testset = self.__generate_testset__() if self.testset is None else self.testset
        assert self.testset is not None, '请实现`_generate_testset`方法，返回测试集'
        return DataLoader(self.testset, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)


class Cifar10Loader(Loader):
    def __init__(self, download=True, root='../data/cifar10', num_workers=1, input_size=None, enhancement=True):
        Loader.__init__(self, root, num_workers, input_size, download, enhancement)

    def __generate_train_transforms__(self):
        train_composes = []
        if self.enhancement:
            train_composes.append(transforms.RandomCrop(32, padding=4))
        if self.input_size is not None:
            train_composes.append(transforms.Resize(self.input_size))
        if self.enhancement:
            train_composes.append(transforms.RandomHorizontalFlip())
        train_composes.append(transforms.ToTensor())
        train_composes.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        return transforms.Compose(train_composes)

    def __generate_test_transforms__(self):
        test_composes = []
        if self.input_size is not None:
            test_composes.append(transforms.Resize(self.input_size))
        test_composes.append(transforms.ToTensor())
        test_composes.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        return transforms.Compose(test_composes)

    def __generate_trainset__(self):
        return CIFAR10(self.root, True, self.train_transforms, download=self.download)

    def __generate_testset__(self):
        return CIFAR10(self.root, False, self.test_transforms, download=self.download)


class Cifar100Loader(Loader):
    def __init__(self, download=True, root='../data/cifar100', num_workers=1, input_size=None, enhancement=True):
        Loader.__init__(self, root, num_workers, input_size, download, enhancement)

    def __generate_train_transforms__(self):
        train_composes = []
        if self.enhancement:
            train_composes.append(transforms.RandomCrop(32, padding=4))
        if self.input_size is not None:
            train_composes.append(transforms.Resize(self.input_size))
        if self.enhancement:
            train_composes.append(transforms.RandomHorizontalFlip())
        train_composes.append(transforms.ToTensor())
        train_composes.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        return transforms.Compose(train_composes)

    def __generate_test_transforms__(self):
        test_composes = []
        if self.input_size is not None:
            test_composes.append(transforms.Resize(self.input_size))
        test_composes.append(transforms.ToTensor())
        test_composes.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        return transforms.Compose(test_composes)

    def __generate_trainset__(self):
        return CIFAR100(self.root, True, self.train_transforms, download=self.download)

    def __generate_testset__(self):
        return CIFAR100(self.root, False, self.test_transforms, download=self.download)


class ILSVRC2012Loader(Loader):
    def __init__(self, root='../data/ILSVRC2012/data/Data/CLS-LOC', num_workers=8, input_size=(224, 224), enhancement=True, tencrop=False):
        self.tencrop = tencrop
        Loader.__init__(self, root, num_workers, input_size, False, enhancement)

    def __generate_train_transforms__(self):
        train_composes = []
        if self.enhancement:
            train_composes.append(transforms.RandomHorizontalFlip(p=0.5))
            train_composes.append(transforms.Resize(self.input_size))
            train_composes.append(transforms.RandomCrop(self.input_size, padding=int(min(self.input_size) / 8)))
            train_composes.append(transforms.RandomRotation(30))
            train_composes.append(transforms.ColorJitter(0.05, 0.05, 0.05, 0.05))
        else:
            train_composes.append(transforms.Resize(self.input_size))
        train_composes.append(transforms.ToTensor())
        train_composes.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        if self.tencrop:
            train_composes.append(transforms.Resize([int(s / 9 * 10) for s in self.input_size]))
            train_composes.append(transforms.TenCrop(self.input_size))
            train_composes.append(transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])))
        return transforms.Compose(train_composes)

    def __generate_test_transforms__(self):
        test_composes = []
        test_composes.append(transforms.Resize(self.input_size))
        test_composes.append(transforms.ToTensor())
        test_composes.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        return transforms.Compose(test_composes)

    def __generate_trainset__(self):
        return ImageFolder('%s/train' % self.root, transform=self.train_transforms)

    def __generate_testset__(self):
        return ImageFolder('%s/val' % self.root, transform=self.test_transforms)
