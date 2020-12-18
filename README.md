# My implementation of ResNet for CIFAR10 in pytorch.

Availabel networks include (trained with 100 epochs):

|   Name   | #Layers | Test top1 Acc(%) |
|----------|---------|------------------|
|ResNet20  | 20      |     90.7         |
|ResNet34  | 34      |     90.9         |
|ResNet74  | 74      |     89.6         |
|ResNet146 | 146     |     89.7         |

## Summary

Residual learning overcomes the degradation problem by providing shortcut path. (My opinion) On the one hand, in the forward path, such shortcut path will replace the redundant layers with identity path, which lead to that the deeper networks be at least as good as shallower ones. On the other hand, in the backward path, shortcut path provide the gradient a way to path a stronger signal to previous layers, which speeds up training.

## Training all ResNet and PlainNet
```
chmode +x run.sh && ./run_all_res_plain_net.sh
```

![alt text](https://github.com/ZXP-S-works/ResNet_CIFAR10/blob/master/figure/comp_plain_res_net.png)

## Training ResNet with layer gradient information
```
chmode +x run.sh && ./run_layer_gradient.sh
```

![alt text](https://github.com/ZXP-S-works/ResNet_CIFAR10/blob/master/figure/plain34layer_gradient.png)
![alt text](https://github.com/ZXP-S-works/ResNet_CIFAR10/blob/master/figure/resnet34layer_gradient.png)

## Training all ResNet shortcut options
```
chmode +x run.sh && ./run_shortcut_options.sh
```

![alt text](https://github.com/ZXP-S-works/ResNet_CIFAR10/blob/master/figure/comp_res_net_option.png)

Visualizations are included in python script visualization.py.

## Reference
1. Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
2. Proper ResNet Implementation for CIFAR10/CIFAR100 in Pytorch (https://github.com/akamaster/pytorch_resnet_cifar10)