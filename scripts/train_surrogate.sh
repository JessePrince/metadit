accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision no \
    train_surrogate.py \
        --num_epoch 500 \
        --batch_size 512 \
        --warmup_ratio 0.002 \
        --optimizer adamw \
        --lr 1e-3 \
        --weight_decay 0.05 \
        --data_path /root/autodl-tmp/colornet/split_data/train_set.mat \
        --val_path /root/autodl-tmp/colornet/split_data/val_set.mat \
        --num_workers 2 \
        --use_checkpointing True \
        --save_dir "ckpt/surrogate-mlphead-s3-2" \
        --save_total_limit 3
