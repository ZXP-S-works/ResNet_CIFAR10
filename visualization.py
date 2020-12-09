import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import networks
import joypy
import pandas as pd


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
    plain_nets = list(arch for arch in  networks.__dict__ if arch.startswith('plain'))
    print(plain_nets)
    for arch in plain_nets:
        all_train_hist[arch] = torch.load('./Results/' + str(arch) + '/model.th')['train_hist'].T

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'wspace': 0})
    fig.suptitle('Learning curve')
    for arch in plain_nets:
        ax1.plot(all_train_hist[arch][1, :], label=arch + ' train acc')
        ax2.plot(all_train_hist[arch][3, :], label=arch + ' test acc')
    ax1.set(ylabel='Top1 accuracy (%)')
    ax1.set(xlabel='epochs')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y')
    ax2.set(xlabel='epochs')
    ax2.legend(loc='lower right')
    ax2.grid(axis='y')
    plt.savefig('./Results/degradation', bbox_inches='tight')
    plt.show()

def comp_plain_res_net():
    # Loading
    all_train_hist = {}
    all_nets = list(arch for arch in  networks.__dict__ if arch.startswith('res') or arch.startswith('plain'))
    res_nets = list(arch for arch in  networks.__dict__ if arch.startswith('res'))
    plain_nets = list(arch for arch in  networks.__dict__ if arch.startswith('plain'))
    for arch in all_nets:
        all_train_hist[arch] = torch.load('./Results/' + str(arch) + '/model.th')['train_hist'].T

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'wspace': 0})
    fig.suptitle('Learning curve')
    for res_net in res_nets:
        color = next(ax1._get_lines.prop_cycler)['color']
        ax1.plot(all_train_hist[res_net][1, :], label=res_net, linestyle='-', color=color)
        ax1.plot(all_train_hist[res_net][3, :], linestyle='-.', color=color)
    for plain_net in plain_nets:
        color = next(ax2._get_lines.prop_cycler)['color']
        ax2.plot(all_train_hist[plain_net][1, :], label=plain_net, linestyle='-', color=color)
        ax2.plot(all_train_hist[plain_net][3, :], linestyle='-.', color=color)
    ax1.set(ylabel='Top1 accuracy (%)')
    ax1.set(xlabel='epochs')
    ax1.set_ylim(55, 95)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y')
    ax2.set(xlabel='epochs')
    ax2.set_ylim(55, 95)
    ax2.legend(loc='lower right')
    ax2.grid(axis='y')
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
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(all_train_hist[option][1, :], label='option ' + option + ' train acc', linestyle='-', color=color)
        ax.plot(all_train_hist[option][3, :], label='option ' + option + ' test acc', linestyle='-.', color=color)
    ax.set(ylabel='Top1 accuracy (%)')
    ax.set(xlabel='epochs')
    ax.set_ylim(55, 95)
    ax.legend(loc='lower right')
    ax.grid(axis='y')
    plt.savefig('./Results/comp_res_net_option', bbox_inches='tight')
    plt.show()

def plot_layer_responses():
    # Loading
    all_activation = []
    # all_nets = list(arch for arch in  networks.__dict__ if arch.startswith('res') or arch.startswith('plain'))
    res_nets = list(arch for arch in  networks.__dict__ if arch.startswith('res'))
    plain_nets = list(arch for arch in  networks.__dict__ if arch.startswith('plain'))
    for (arch1, arch2) in zip(res_nets, plain_nets):
        dict1 = torch.load('./Results/' + str(arch1) + '/model.th')['activation']
        list1 = (arch1, list(dict1.values()))
        dict2 = torch.load('./Results/' + str(arch2) + '/model.th')['activation']
        list2 = (arch2, list(dict2.values()))

        all_activation.append([list1, list2])

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 6))
    fig.suptitle('Layer Responses')
    for item in all_activation:
        color = next(ax1._get_lines.prop_cycler)['color']
        ax1.plot(item[0][1], label=item[0][0], linestyle='-', color=color)
        ax1.plot(item[1][1], label=item[1][0], linestyle=':', color=color)
    for item in all_activation:
        color = next(ax2._get_lines.prop_cycler)['color']
        ax2.plot(sorted(item[0][1], reverse=True), label=item[0][0], linestyle='-', color=color)
        ax2.plot(sorted(item[1][1], reverse=True), label=item[1][0], linestyle=':', color=color)
    ax1.set(ylabel='std')
    ax1.set(xlabel='layers (original)')
    ax1.set_xlim(0, 150)
    ax1.legend(loc='upper right')
    ax2.set(ylabel='std')
    ax2.set(xlabel='layers (sorted)')
    ax2.legend(loc='upper right')
    plt.savefig('./Results/layer_activation', bbox_inches='tight')
    plt.show()

def plot_layer_gradient():
    # Loading
    all_gradient_hist = {}
    all_models = ['resnet34', 'plain34']
    for models in all_models:
        all_gradient_hist[option] = torch.load('./Results/' + str(models) + '/model.th')['gradient_hist']

    # to panda data frame
    epochs = range(1, len(all_gradient_hist[all_models[0]])+1)
    df1 = pd.DataFrame(list(zip(all_gradient_hist[all_models[0]], epochs)), columns=['Epoch', 'Gradient Norm'])
    df2 = pd.DataFrame(list(zip(all_gradient_hist[all_models[1]], epochs)), columns=['Epoch', 'Gradient Norm'])

    # Plots
    fig1, axe1 = joypy.joyplot(df1, by="Epoch", column="Gradient Norm", figsize=(5, 8))
    fig2, axe2 = joypy.joyplot(df2, by="Epoch", column="Gradient Norm", figsize=(5, 8))

    # fig, ax = plt.subplots(1)
    # fig.suptitle('Learning curve')
    # for option in all_options:
    #     color = next(ax1._get_lines.prop_cycler)['color']
    #     ax.plot(all_train_hist[option][1, :], label='option ' + option + ' train acc', linestyle='', color=color)
    #     ax.plot(all_train_hist[option][3, :], label='option ' + option + ' test acc', linestyle=':', color=color)
    # ax.set(ylabel='Top1 accuracy (%)')
    # ax.set(xlabel='epochs')

    plt1.savefig('./Results/' + all_models[0] + 'layer_gradient', bbox_inches='tight')
    plt2.savefig('./Results/' + all_models[1] + 'layer_gradient', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization and statistics of the trained ResNet')
    parser.add_argument('--dir', '--directory', metavar='DIRE', type=str,
                        help='The directory that saves the trained model')
    arg = parser.parse_args()

    visualization(arg.dir)
