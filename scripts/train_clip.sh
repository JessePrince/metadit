accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train/train_clip.py \
        --num_epoch 500 \
        --batch_size 512 \
        --warmup_ratio 0.0 \
        --optimizer adamw \
        --lr 2e-5 \
        --weight_decay 0.0 \
        --data_path "split_data/train_set.mat" \
        --val_path split_data/val_set.mat \
        --num_workers 2 \
        --use_checkpointing True \
        --save_dir "ckpt/clip-final-distributed-loss" \
        --high_res_spec True \
        --condition_channel 301 \
