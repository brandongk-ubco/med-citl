#!/usr/bin/env bash

export $(cat .env | xargs)

# nohup ./cifar.sh > cifar.log 2>&1 &
nohup ./puma.sh > puma.log 2>&1 &
# nohup ./cityscapes.sh > cityscapes.log 2>&1 &