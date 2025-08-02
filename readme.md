## MetaDiT: Enabling Fine-grained Constraints in High-degree-of Freedom Metasurface Design

Official code implementation of paper **MetaDiT: Enabling Fine-grained Constraints in High-degree-of Freedom Metasurface Design**.

### News
[2025.08.02] 🔥 We release the first version of our code! We will add more comments and optimize the code structure in the future!

### Setups
1. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```
2. Download dataset from https://github.com/SensongAn/Meta-atoms-data-sharing. You can split the dataset to train/val/test set by yourself or use our split version.

### Training
To train the surrogate model
```bash
bash scripts/train_surrogate.sh
```

The arguments include
- `--num_epoch` Number of epochs to train
- `--batch_size` batch size for training
- `--warmup_ratio` warmup ratio for learning rate
- `--optimizer` optimizer to use, supports AdamW only
- `--lr` learning rate
- `--weight_decay` weight decay used in the optimizer
- `--data_path` path to train data (.mat)
- `--val_path` path to validation data (.mat)
- `--num_workers` number of workers for dataloader
- `--use_checkpointing` Whether to use gradient checkpointing
- `--save_dir` path to save
- `--save_total_limit` maximum save limit

More arguments can be found in `train/train_surrogate.py`.

To train the Spectrum Encoder using CLIP
```bash
bash scripts/train_clip.sh
```

The arguments include
- `--num_epoch` Number of epochs to train
- `--batch_size` batch size for training
- `--warmup_ratio` warmup ratio for learning rate
- `--optimizer` optimizer to use, supports AdamW only
- `--lr` learning rate
- `--weight_decay` weight decay used in the optimizer
- `--data_path` path to train data (.mat)
- `--val_path` path to validation data (.mat)
- `--num_workers` number of workers for dataloader
- `--use_checkpointing` Whether to use gradient checkpointing
- `--save_dir` path to save
- `--high_res_spec` Whether to use high resolution spectrum
- `--condition_channel` channel of the spectrum

More arguments can be found in `train/train_clip.py`.

To train the MetaDiT model
```bash
bash scripts/train_metadit.sh
```

The arguments include
- `--num_epoch` Number of epochs to train
- `--batch_size` batch size for training
- `--warmup_ratio` warmup ratio for learning rate
- `--optimizer` optimizer to use, supports AdamW only
- `--lr` learning rate
- `--weight_decay` weight decay used in the optimizer
- `--data_path` path to train data (.mat)
- `--val_path` path to validation data (.mat)
- `--num_workers` number of workers for dataloader
- `--use_checkpointing` Whether to use gradient checkpointing
- `--save_dir` path to save
- `--high_res_spec` Whether to use high resolution spectrum
- `--condition_channel` channel of the spectrum
- `--pretrain_encoder` path to the pretrained encoder
        

### Inference
To sample material from MetaDiT on the test set
```bash
bash scripts/test_metadit.sh
```

The arguments include
- `--seed` random seed used in inference
- `--ckpt` path to the MetaDiT checkpoint
- `--num_sampling_steps` number of diffusion sampling steps
- `--data_path` path to the test set
- `--batch_size` batch size used in sampling
- `--resolution` resolution of the spectrum
- `--save_path` path to save the sampling results
- `--high_res_spec` whether to use high resolution spectrum.

We support distributed inference (DDP inference), if you have multiple GPUs, please set the corresponding `--num_proccesses` when you use `accelerate launch`.

To calculate the AAE and AAE&K score
```bash
python calc_aae.py --data_path <path to the sampling results> --model_path <path to the surrogate model>
```
The folder structure should be 
- Sampling results
  - xxx_rank0.json
  - xxx_rank1.json
  - ...


To calculate AAE&K
```bash
python calc_aaeandk.py --data_path <path to the sampling results> --model_path <path to the surrogate model>
```

The folder structure should be
- Sampling results
  - xxx_seed0
    - xxx_rank0.json
    - xxx_rank1.json
    - ...
  - xxx_seed7
    - xxx_rank0.json
    - xxx_rank1.json
    - ...
  - xxx_seed42
    - xxx_rank0.json
    - xxx_rank1.json
    - ...
  - xxx_seed3407
    - xxx_rank0.json
    - xxx_rank1.json
    - ...

### Accelerators
We suggest at least 4 GPUs with more than 24GB memory, we used 4$\times$ Nvidia A100 80GB in this project.
