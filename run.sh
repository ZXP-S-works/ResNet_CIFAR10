#!/bin/bash

for arch in resnet20 resnet34 resnet74 resnet146 plain20 plain34 plain74 plain146
do
    echo "python3 train_test.py --arch=$arch |& tee -a ./Results/$arch/log"
    python3 train_test.py --arch=$arch |& tee -a ./Results/$arch/log
done
