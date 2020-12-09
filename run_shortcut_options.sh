#!/bin/bash

for option in A B C D
do
    echo "python3 train_test.py --option=$option |& tee -a ./Results/resnet34_$option/log"
    python3 train_test.py --option=$option |& tee -a ./Results/resnet34_$option/log
done
