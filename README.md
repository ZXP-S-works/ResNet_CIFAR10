# My implementation of ResNet for CIFAR10 in pytorch.

Availabel networks include:

|   Name   | #layers |
|----------|---------|
|ResNet20  | 20      |
|ResNet34  | 34      |
|ResNet74  | 74      |
|ResNet146 | 146     |

## Training all ResNet and PlainNet
```
chmode +x run.sh && ./run_all_res_plain_net.sh
```


## Training ResNet with layer gradient information
```
chmode +x run.sh && ./run_all_res_plain_net.sh
```


## Training all ResNet shortcut options
```
chmode +x run.sh && ./run_all_res_plain_net.sh
```

Visualizations are included in python script visualization.py.
