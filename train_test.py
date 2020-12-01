import numpy as np
import torch
import torch.utils.data
import networks
import utils
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from parameters import *


def main():
    # Initialize the network
    ResNet = networks.__dict__[args.arch]().to(device)
    print(ResNet)

    # Initialize train/test set
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='edge'),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          normalize])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         normalize])
    train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    # Initialize torch gradient setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ResNet.parameters(), lr=args.lr, weight_decay=args.wd)
    milestones = np.linspace(0, args.epochs, args.milestones+2)[1:-1]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay)

    # Initialize statistics for training
    # epoch_time = []
    # loading_time = []
    train_hist = []

    # Train-test
    print('Learning rate: {}'.format(args.lr))
    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch, ResNet, train_loader, criterion, optimizer)
        scheduler.step()
        print('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        test_loss, test_acc = test(ResNet, test_loader, criterion)
        train_hist.append([train_loss, train_acc, test_loss, test_acc])

    train_hist = np.array(train_hist)

    # Plots
    plt.figure(figsize=(12, 8))
    plt.plot(train_hist)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title('Learning curve', fontsize=24)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('NLL', fontsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.show()


def train(epoch, model, train_loader, criterion, optimizer):
    """
    One train epoch
    """
    # initialize parameters for statistics
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()

    tic = time.time()
    for i, (img, labels) in enumerate(train_loader):
        # data loading time
        data_time.update(time.time() - tic)

        # to cuda if possible else cpo
        img = img.to(device)
        labels = labels.to(device)

        # back propagation
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # statistics of a mini-batch SGD
        loss = loss.cpu().item()
        output = output.cpu()
        labels = labels.cpu()
        precision1 = utils.accuracy(output, labels)[0]
        losses.update(loss)
        top1.update(precision1, labels.shape[0])
        batch_time.update(time.time() - tic)
        tic = time.time()

    # Print statistics
    print('Epoch[{0}]\tLoading Time: {1:.3f}\tEpoch Time: {2:.3f}\tLoss: {3:.4f}\tAccuracy@1: {4:.3f}'
          .format(epoch+1, data_time.sum, batch_time.sum, losses.avg, top1.avg))

    return losses.avg, top1.avg


def test(model, test_loader, criterion):
    return 0, 0


if __name__ == '__main__':
    main()
