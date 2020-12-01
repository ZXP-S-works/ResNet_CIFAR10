import argparse
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# parameters parsing
parser = argparse.ArgumentParser(description='Training parameters for ResNet in CIFAR10')
parser.add_argument('--arch', '--a', '--architecture', metavar='ARCH', default='resnet20',
                    help='model architecture')
parser.add_argument('--bs', '--batch-size', metavar='N', default='128', type=int,
                    help='batch size for training')
parser.add_argument('--workers', metavar='N', default=1, type=int,
                    help='number of workers in data loading')
parser.add_argument('--lr', '--learning-rate', metavar='R', default='0.01', type=float,
                    help='learning rate for the SGD')
parser.add_argument('--wd', '--weight-decay', metavar='R', default=0.01,
                    help='L2 penalty for parameters regularization')
parser.add_argument('--milestones', metavar='N', default=5, type=int,
                    help='number of milestones, milestones will be evenly set')
parser.add_argument('--lr-decay', metavar='R', default=0.1, type=float,
                    help='decay on learning rate when a milestone is reached')
parser.add_argument('--epochs', metavar='N', default=2000, type=int,
                    help='number of epochs for training')

args = parser.parse_args()

# Print all parameters
var_args = vars(args)
for key in var_args:
    print(key, var_args[key])
