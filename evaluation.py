import torch
import numpy as np
import networks
from collections import defaultdict
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
test_transform = transforms.Compose([transforms.ToTensor(),
                                     normalize])
test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4)


def calcu_layer_responses(directory):
    # Loading
    model_dict = torch.load(directory + '/model.th')
    state_dict = model_dict['state_dict']
    arch = directory.split('/')[-1].split('_')[0]
    model = networks.__dict__[arch]().to(device)
    model.load_state_dict(state_dict)

    # hook for getting activation std
    activation = defaultdict(lambda : 0)
    def get_activation_std(name):
        def hook(model, input, output):
            activation[name] += output.data.std(dim=1).mean().item()  # there may be some bugs
            # print(output.data.shape)
        return hook

    # calculate activation std for each layer on the whole test set
    model.eval()
    with torch.no_grad():
        for i, (img, _) in enumerate(test_loader):
            # forward propagation
            img = img.to(device)
            output = model(img)

            # get activation std
            for name, layer in model.named_modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.register_forward_hook(get_activation_std(name))

        for item in activation:
            activation[item] /= len(test_loader)
        print(activation)

    return activation
