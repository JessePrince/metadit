accelerate launch test/test_metadit.py \
    --seed 0 \
    --ckpt /root/autodl-tmp/colornet/ckpt/metadit-cond-ffn-2/epoch_500/pytorch_model.bin \
    --num_sampling_steps 500 \
    --data_path split_data/test_set.mat \
    --batch_size 256 \
    --resolution 32 \
    --save_path test_results/metadit-cond-ffn-2 \
    --high_res_spec True \
