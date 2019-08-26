import framework
import torch
from torchvision import models
from torch import nn

DEVICE = torch.device("cuda:1,2,3,4,5,6,7" if torch.cuda.is_available() else "cpu")
alexnet = models.AlexNet(num_classes=10)
mul_alexnet = nn.DataParallel(alexnet, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])


# NN.train_schedule('mutiple_train', mul_alexnet, epochs=[2, 2, 2], lrs=[0.001, 0.0005, 0.0001], save_interval=10)
framework.evaluate(alexnet, 'mutiple_train.pkl', DEVICE=DEVICE)
