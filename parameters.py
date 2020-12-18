import argparse
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# parameters parsing
parser = argparse.ArgumentParser(description='Training parameters for ResNet in CIFAR10')
parser.add_argument('--arch', '--a', '--architecture', metavar='ARCH', default='resnet34',
                    help='model architecture')
parser.add_argument('--bs', '--batch-size', metavar='N', default=128, type=int,
                    help='batch size for training')
parser.add_argument('--workers', metavar='N', default=4, type=int,
                    help='number of workers in data loading')
parser.add_argument('--lr', '--learning-rate', metavar='R', default=0.016, type=float,
                    help='learning rate for the SGD')
parser.add_argument('--wd', '--weight-decay', metavar='R', default=1e-4, type=float,
                    help='L2 penalty for parameters regularization')
parser.add_argument('--milestones', metavar='N', default=2, type=int,
                    help='number of milestones, milestones will be evenly set')
parser.add_argument('--lr-decay', metavar='R', default=0.5, type=float,
                    help='decay on learning rate when a milestone is reached')
parser.add_argument('--epochs', metavar='N', default=100, type=int,
                    help='number of epochs for training')
parser.add_argument('--gn', '--gradient-norm', metavar='B', default=False, type=bool,
                    help='whether record gradient norm during train or not')
parser.add_argument('--option', metavar='option', default='default', type=str, choices=['defalut', 'A', 'B', 'C', 'D'],
                    help='resudual block shotcut option')
parser.add_argument('--dir', '--directory', metavar='PATH', default='default', type=str,
                    help='directory for saving the model')

args = parser.parse_args()


# Print all parameters
var_args = vars(args)
for key in var_args:
    print(key, var_args[key])
