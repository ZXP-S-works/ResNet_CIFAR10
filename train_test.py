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
import os


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
    milestones = np.linspace(0, args.epochs, args.milestones+2)[1:-1].astype(int).tolist()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay)

    # Initialize statistics for training
    # epoch_time = []
    # loading_time = []
    train_hist = []

    # Train-test
    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch, ResNet, train_loader, criterion, optimizer)
        scheduler.step()
        test_loss, test_acc = test(ResNet, test_loader, criterion)
        train_hist.append([train_loss, train_acc, test_loss, test_acc])

    # statistics and save checkpoint
    train_hist = np.array(train_hist)
    for i in range(100):
        save_dir = './Results/' + args.arch + '_' + str(i+1)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
    utils.save_checkpoint({'state_dict': ResNet.state_dict(),
                           'train_hist': train_hist},
                          filename=os.path.join(save_dir, '/model.th'))

    # Plots
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(train_hist[1, 3])
    # plt.xlim(left=0)
    # plt.ylim(bottom=0)
    fig.title('Learning curve', fontsize=24)
    ax1.xlabel('Epochs', fontsize=16)
    ax1.ylabel('Top1 accuracy (%)', fontsize=16)
    ax1.legend(['Train acc', 'Test acc'], loc='upper right', fontsize=14)
    ax2.plot(train_hist[1, 3])
    # plt.xlim(left=0)
    # plt.ylim(bottom=0)
    ax2.xlabel('Epochs', fontsize=16)
    ax2.ylabel('Loss (NNL)', fontsize=16)
    ax2.legend(['Train loss', 'Test loss'], loc='upper right', fontsize=14)
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
    print('Epoch[{0}]\tTrain:\t'
          'Accuracy@1: {1:.2f}\t'
          'Loss: {2:.4f}\t'
          'Loading Time: {3:.2f}\t'
          'Epoch Time: {4:.2f}\t'
          'Learning Rate: {5:.2e}'
          .format(epoch+1,
                  top1.avg,
                  losses.avg,
                  data_time.sum,
                  batch_time.sum,
                  optimizer.param_groups[0]['lr']))

    return losses.avg, top1.avg


def test(model, test_loader, criterion):
    """
        One test epoch
    """
    # initialize parameters for statistics
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()

    tic = time.time()
    with torch.no_grad():
        for i, (img, labels) in enumerate(test_loader):
            # data loading time
            data_time.update(time.time() - tic)

            # to cuda if possible else cpo
            img = img.to(device)
            labels = labels.to(device)

            # back propagation
            output = model(img)
            loss = criterion(output, labels)

            # statistics of a mini-batch
            loss = loss.cpu().item()
            output = output.cpu()
            labels = labels.cpu()
            precision1 = utils.accuracy(output, labels)[0]
            losses.update(loss)
            top1.update(precision1, labels.shape[0])
            batch_time.update(time.time() - tic)
            tic = time.time()

    # Print statistics
    print('\t\t\tTest:\t'
          'Accuracy@1: {0:.2f}\t'
          'Loss: {1:.4f}\t'
          'Loading Time: {2:.2f}\t'
          'Epoch Time: {3:.2f}'
          .format(top1.avg,
                  losses.avg,
                  data_time.sum,
                  batch_time.sum))

    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
