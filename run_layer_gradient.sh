#!/bin/bash

for arch in plain34 resnet34
do
    echo "python3 train_test.py --arch $arch --gn True |& tee -a ./Results/$arch_gradient/log"
    python3 train_test.py --arch $arch --gn True |& tee -a ./Results/$arch_gradient/log
done
