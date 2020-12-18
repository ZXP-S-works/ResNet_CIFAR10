#!/bin/bash

for lr in 0.1 0.05 0.01
do
    echo "python3 train_test.py --arch=resnet20 --epochs 50 --lr $lr --dir /Results/resnet20_lr$lr"
    python3 train_test.py --arch=resnet20 --epochs 50 --lr $lr --dir /Results/resnet20_lr$lr
done
