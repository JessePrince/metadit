"""
Calculate the MAE of the surrogate model on test set
"""

import torch
import torch.nn.functional as F

from datapipe import SurrogateFreeFormDataset
from model.surrogate import surrogate_s3
from argparse import ArgumentParser
from torch.utils.data import DataLoader




def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/colornet/ckpt/surrogate-mlphead-s3-2/epoch_500/pytorch_model.bin")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/colornet/split_data/test_set.mat")
    parser.add_argument("--batch_size", type=int, default=2048)
    
    args = parser.parse_args()
    
    return args


def main(args):
    assert torch.cuda.is_available()
    dataset = SurrogateFreeFormDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model = surrogate_s3()
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt, strict=True)
    model.to("cuda")
    model.eval()
    total_samples = 0
    total_mae = 0
    with torch.no_grad():
        for data in loader:
            B = data["inputs"].shape[0]
            total_samples += B
            for k, v in data.items():
                data[k] = v.to("cuda")
                
            output = model(**data)
            print("Loss for this batch:", output.loss.item())
            total_mae += output.loss.item() * B
            
    average_mae = total_mae / total_samples
    print("Average MAE:", average_mae)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)