#!/bin/bash

seeds=(0 7 42 3407)

for seed in "${seeds[@]}"; do
    python generate.py \
        --num_gpus 4 \
        --temp_dir cache/inference \
        --data_path /root/autodl-tmp/metadit/sim_data/test_set.mat \
        --model_path /root/autodl-tmp/metadit/ckpts/metadit-small.bin \
        --model_type metadit_s \
        --condition_channel 301 \
        --seed "$seed" \
        --resolution 32 \
        --cfg_scale 4.0 \
        --batch_size 256 \
        --time_steps 500 \
        --save_path "results/seed${seed}.json"
done
