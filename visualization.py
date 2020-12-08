import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import networks


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
    # plt.show()

def degradation():
    # Loading
    all_train_hist = {}
    plain_nets = sorted(arch for arch in resnet.__dict__ if name.startswith('plain'))
    for arch in plain_nets:
        all_train_hist[arch] = torch.load('./Results/' + str(arch) + '/model.th')['train_hist'].T

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle('Learning curve')
    for arch in plain_nets:
        ax1.plot(all_train_hist[arch][1, :], label=arch + ' train acc')
        ax2.plot(all_train_hist[arch][3, :], label=arch + ' test acc')
    ax1.set(ylabel='Top1 accuracy (%)')
    ax1.set(xlabel='epochs')
    ax2.set(xlabel='epochs')
    plt.savefig('./Results/degradation', bbox_inches='tight')
    plt.show()


def comp_plain_res_net():
    # Loading
    all_train_hist = {}
    all_nets = sorted(arch for arch in resnet.__dict__)
    res_nets = sorted(arch for arch in resnet.__dict__ if name.startswith('res'))
    plain_nets = sorted(arch for arch in resnet.__dict__ if name.startswith('plain'))
    for arch in all_nets:
        all_train_hist[arch] = torch.load('./Results/' + str(arch) + '/model.th')['train_hist'].T

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle('Learning curve')
    for res_net in res_nets:
        color = next(ax1._get_lines.prop_cycler)['color']
        ax1.plot(all_train_hist[res_net][1, :], label=res_net + ' train acc', linestyle='', color=color)
        ax1.plot(all_train_hist[res_net][3, :], label=res_net + ' test acc', linestyle='_', color=color)
    for plain_net in plain_nets:
        color = next(ax2._get_lines.prop_cycler)['color']
        ax2.plot(all_train_hist[plain_net][1, :], label=plain_net + ' train acc', linestyle='', color=color)
        ax2.plot(all_train_hist[plain_net][3, :], label=plain_net + ' test acc', linestyle='_', color=color)
    ax1.set(ylabel='Top1 accuracy (%)')
    ax1.set(xlabel='epochs')
    ax2.set(xlabel='epochs')
    plt.savefig('./Results/comp_plain_res_net', bbox_inches='tight')
    plt.show()


def comp_res_net_option():
    # Loading
    all_train_hist = {}
    all_options = ['A', 'B', 'C', 'D']
    for option in all_options:
        all_train_hist[option] = torch.load('./Results/resnet34_' + str(option) + '/model.th')['train_hist'].T

    # Plots
    fig, ax = plt.subplots(1)
    fig.suptitle('Learning curve')
    for option in all_options:
        color = next(ax1._get_lines.prop_cycler)['color']
        ax.plot(all_train_hist[option][1, :], label='option ' + option + ' train acc', linestyle='', color=color)
        ax.plot(all_train_hist[option][3, :], label='option ' + option + ' test acc', linestyle='_', color=color)
    ax.set(ylabel='Top1 accuracy (%)')
    ax.set(xlabel='epochs')
    plt.savefig('./Results/comp_res_net_option', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization and statistics of the trained ResNet')
    parser.add_argument('--dir', '--directory', metavar='DIRE', type=str,
                        help='The directory that saves the trained model')
    arg = parser.parse_args()

    visualization(arg.dir)
