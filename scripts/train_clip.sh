accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train/train_clip.py \
        --num_epoch 500 \
        --batch_size 256 \
        --warmup_ratio 0.0 \
        --optimizer adamw \
        --lr 5e-4 \
        --weight_decay 0.0 \
        --data_path "/root/autodl-tmp/colornet/split_data/train_set.mat" \
        --val_path /root/autodl-tmp/colornet/split_data/val_set.mat \
        --num_workers 2 \
        --use_checkpointing True \
        --save_dir "ckpt/clip-final-distributed-loss" \
        --high_res_spec True \
        --condition_channel 602 \
        
