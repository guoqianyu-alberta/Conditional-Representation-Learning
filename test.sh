#!/bin/bash
python test.py \
        --n_shot 1 \
        --backbone 'Resnet12' \
        --pretrained_path 'pretrained/res12.tar' \
        --load 'checkpoints/resnet12.pt' \
        --eval_dataset plant_virus \
        --eval_episodes 10 \
        --n_query 15 \
        --image_size 224 \
        --gpu 5


