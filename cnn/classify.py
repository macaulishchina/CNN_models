# -*- coding: utf-8 -*-
import time
import math
import json
import torch
from torch import nn
from torch import optim
import progressbar as pb
from framework import visualize_feedback


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            bar.currval = 0
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
         dataLoader,
         dataset_type='',
         print_feedback=True,
         batch_size=128,
         DEVICE=DEVICE
         ):
    model.to(DEVICE)
    model.eval()
    epoch_accuracy = 0
    batch_count = 0
    print('Testing [%s], sum=%s, batch_size=%s ...' % (dataset_type, len(dataLoader.dataset), batch_size))
    weights = [pb.Percentage(), '(', pb.SimpleProgress(), ')', pb.Bar('>'), '[accuracy=%0.3f%%]' % (0), pb.ETA()]
    with pb.ProgressBar(max_value=math.ceil(len(dataLoader.dataset) / batch_size), widgets=weights) as bar:
        bar.currval = 0
        for inputs, labels in dataLoader:
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
                   dataLoader,
                   loss_F=None,
                   optimizer=None,
                   epochs='10',
                   lrs='0.0001',
                   batch_size=512,
                   trainable=None,
                   DEVICE=DEVICE,
                   save_interval=10,
                   save_feedback=True,
                   print_feedback=True,
                   weights_dir='./weights'
                   ):
    last_epoch = 0
    feedback = {}
    epochs, lrs = epochs.split(','), lrs.split(',')
    assert len(epochs) == len(lrs)
    for idx, lr in enumerate(lrs):
        epoch, lr = int(epochs[idx]), float(lr)
        model, fb = train(model, dataLoader,
                          tag=tag, batch_size=batch_size, DEVICE=DEVICE,
                          epochs=epoch, lr=lr, trainable=trainable,
                          save_interval=save_interval, optimizer=optimizer,
                          save_dir=weights_dir, last_epoch=last_epoch)
        last_epoch += epoch
        feedback.update(fb)
    weights_file = '%s/%s.pkl' % (weights_dir, tag)
    save_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(save_dict, weights_file)

    feedback_file = '%s/[%s]%s.json' % (weights_dir, tag, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    with open(feedback_file, 'w') as file:
        json.dump(feedback, file)
    visualize_feedback(feedback=feedback, save_dir=weights_dir, tag=tag)
    return model, feedback


def evaluate(model, dataLoader, batch_size=512, DEVICE=DEVICE, contain_train=False):
    if contain_train:
        trainset = dataLoader.get_train_loader(batch_size=batch_size, shuffle=False)
        test(model, trainset, 'trainset', batch_size=batch_size, DEVICE=DEVICE)
    testset = dataLoader.get_test_loader(batch_size=batch_size, shuffle=False)
    test(model, testset, 'testset', batch_size=batch_size, DEVICE=DEVICE)