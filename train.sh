#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=4 --master_port=22216 train.py \
                    --image_size 224 \
                    --train_bsz 80 \
                    --epochs 100 \
                    --backbone "Resnet12" \
                    --train_dataset "ImageNet"  \
                    --lr 1e-3  