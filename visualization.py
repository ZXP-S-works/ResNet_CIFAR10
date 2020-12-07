import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse


def visualization(directory):
    # Loading
    model_dict = torch.load(directory + '/model.th')
    train_hist = model_dict['train_hist'].T

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle('Learning curve')
    ax1.plot(train_hist[1, :])
    ax1.plot(train_hist[3, :])
    ax1.set(ylabel='Top1 accuracy (%)')
    ax1.legend(['Train acc', 'Test acc'], loc='lower right')
    ax1.grid()
    ax2.plot(train_hist[0, :])
    ax2.plot(train_hist[2, :])
    ax2.set(xlabel='Epochs', ylabel='Loss (NNL)')
    ax2.legend(['Train loss', 'Test loss'], loc='upper right')
    ax2.grid()
    plt.savefig(directory + '/learning_curve.png', bbox_inches='tight')
    plt.show()


def degradation(directory1, directory2, layers):
    # Loading
    model_dict1 = torch.load(directory1 + '/model.th')
    train_hist1 = model_dict1['train_hist'].T
    model_dict2 = torch.load(directory2 + '/model.th')
    train_hist2 = model_dict2['train_hist'].T

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle('Learning curve')
    ax1.plot(train_hist[1, :])
    ax1.plot(train_hist[3, :])
    ax1.set(ylabel='Top1 accuracy (%)')
    ax1.legend(['Train acc', 'Test acc'], loc='lower right')
    ax1.grid()
    ax2.plot(train_hist[0, :])
    ax2.plot(train_hist[2, :])
    ax2.set(xlabel='Epochs', ylabel='Loss (NNL)')
    ax2.legend(['Train loss', 'Test loss'], loc='upper right')
    ax2.grid()
    plt.savefig(directory + '/learning_curve.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization and statistics of the trained ResNet')
    parser.add_argument('--dir', '--directory', metavar='DIRE', type=str,
                        help='The directory that saves the trained model')
    arg = parser.parse_args()

    visualization(arg.dir)
