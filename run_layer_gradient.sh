#!/bin/bash

for arch in plain34 resnet34
do
    echo "python3 train_test.py --arch $arch --gn True |& tee -a ./Results/$arch/log"
    python3 train_test.py --arch $arch --gn True |& tee -a ./Results/$arch/log
done
