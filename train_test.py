import numpy as np
import torch
import networks
import utils
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# parameters parsing
parser = argparse.ArgumentParser(description='Training parameters for ResNet in CIFAR10')
parser.add_argument('--arch', '--a', '--architecture', metaver='ARCH', defalut='resnet20',
                    help='model architecture')
parser.add_argument('--bs', '--batch-size', metaver='N', defalut='128', type=int,
                    help='batch size for training')
parser.add_argument('--workers', metavar='N', default=1, type=int,
                    help='number of workers in data loading')
parser.add_argument('--lr', '--learning-rate', metaver='R', defalut='0.01', type=float,
                    help='learning rate for the SGD')
parser.add_argument('--wd', '--weight-decay', metavar='R', default=0.01,
                    help='L2 penalty for parameters regularization')
parser.add_argument('--milestones', metavar='N', default=5, type=int,
                    help='number of milestones, milestones will be evenly set')
parser.add_argument('--lr-decay', metavar='R', default=0.1, type=float,
                    help='decay on learning rate when a milestone is reached')
parser.add_argument('--epochs', metavar='N', default=2000, type=int,
                    help='number of epochs for training')


def main():
    args = parser.parse_args()

    # Initialize the network
    ResNet = networks.__dict__[args.arch]
    ResNet.cuda()

    # Initialize train/test set
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # ImageNet statistics
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          normalize,
                                          transforms.RandomCrop(32, padding=4, padding_mode='edge'),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15)])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                          normalize])
    train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    # Initialize torch gradient setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ResNet.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    # Initialize statistics for training
    # epoch_time = []
    # loading_time = []
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # Train-test
    for epoch in tqdm(range(args.epochs)):
        train_loss, train_acc = train(ResNet, train_loader, criterion, optimizer)
        test_loss, test_acc = test(ResNet, test_loader, criterion)

    # Plots


def train(ResNet, train_loader, criterion, optimizer):
    return loss, acc


def test(ResNet, test_loader, criterion):
    return loss, acc


if __name__ == '__main__':
    main()
