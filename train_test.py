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
import visualization
from torchsummary import summary
import evaluation


def main():
    # Initialize the network
    if args.option == 'default':
        ResNet = networks.__dict__[args.arch]('A').to(device)
    else:
        ResNet = networks.__dict__[args.arch](args.option).to(device)
    print(ResNet)
    summary(ResNet, input_size=(3, 32, 32))

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
    milestones = np.linspace(0, args.epochs, args.milestones + 2)[1:-1].astype(int).tolist()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay)

    # Initialize statistics for training
    # epoch_time = []
    # loading_time = []
    train_hist = []

    # recorde gradient norm
    if args.gn:
        gradient_recorder = utils.GradientNorm(ResNet)
    else:
        gradient_recorder = None

    # Train-test
    tic = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch, ResNet, train_loader, criterion, optimizer, gradient_recorder)
        scheduler.step()
        test_loss, test_acc = test(ResNet, test_loader, criterion)
        train_hist.append([train_loss, train_acc, test_loss, test_acc])
    toc = time.time()
    print('Total traning time: {:.2f}s'.format(toc - tic))

    # recorde gradient norm
    if args.gn:
        gradient_norm_hist = gradient_recorder.get_graidnet_norm_hist()
        gradient_norm_hist = np.array(gradient_norm_hist)/len(train_loader)
    else:
        gradient_norm_hist = None

    # statistics and save checkpoint
    train_hist = np.array(train_hist)
    # for i in range(100):
    #     save_dir = './Results/' + args.arch + '_' + str(i + 1)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #         break

    if args.option == 'defalut':
        save_dir = './Results/' + args.arch
    else:
        save_dir = './Results/' + args.arch + '_' + args.option
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    activation = evaluation.calcu_layer_responses(None, ResNet)
    torch.save({'state_dict': ResNet.state_dict(),
                'train_hist': train_hist,
                'gradient_hist': gradient_norm_hist,
                'activation': activation},
               os.path.join(save_dir, 'model.th'))


    # visualization
    visualization.visualization(save_dir)


def train(epoch, model, train_loader, criterion, optimizer, gradient_recorder=None):
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

        # to cuda if possible else cpu
        img = img.to(device)
        labels = labels.to(device)

        # back propagation
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # recorde gradient norm
        if args.gn:
            gradient_recorder.calcu_gradient_norm()

        # statistics of a mini-batch SGD
        loss = loss.item()
        output = output.cpu()
        labels = labels.cpu()
        precision1 = utils.accuracy(output, labels)[0]
        losses.update(loss)
        top1.update(precision1, labels.shape[0])
        batch_time.update(time.time() - tic)
        tic = time.time()

    # recorde gradient norm
    if args.gn:
        gradient_recorder.update_hist()

    # Print statistics
    print('Epoch[{}/{}]\tTrain:\t'
          'Accuracy@1: {:.2f}\t'
          'Loss: {:.4f}\t'
          'Loading Time: {:.2f}\t'
          'Epoch Time: {:.2f}\t'
          'Learning Rate: {:.2e}'
          .format(epoch + 1, args.epochs,
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

            # to cuda if possible else cpu
            img = img.to(device)
            labels = labels.to(device)

            # forward propagation
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
    print('\t\tTest:\t '
          'Accuracy@1: {0:.2f}\t '
          'Loss: {1:.4f}\t '
          'Loading Time: {2:.2f}\t '
          'Epoch Time: {3:.2f}'
          .format(top1.avg,
                  losses.avg,
                  data_time.sum,
                  batch_time.sum))

    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
