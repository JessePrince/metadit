accelerate launch \
    --num_processes 7 \
    --num_machines 1 \
    --mixed_precision no \
    train/train_metadit.py \
        --num_epoch 500 \
        --batch_size 128 \
        --warmup_ratio 0 \
        --optimizer adamw \
        --lr 1e-4 \
        --weight_decay 0.0 \
        --data_path "split_data/train_set.mat" \
        --val_path split_data/val_set.mat \
        --num_workers 2 \
        --use_checkpointing True \
        --save_dir "ckpt/metadit-self-attn-large" \
        --high_res_spec True \
        --condition_channel 602 \
        --pretrain_encoder spec_encoder.pth \
        
