import os
import torch

from utils import load_json
from model.surrogate import surrogate_s3
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from loggers import WrappedLogger
from tqdm import tqdm

logger = WrappedLogger(__name__)


def main(args):
    files = os.listdir(args.data_path)
    data = []
    for file in files:
        logger.info(f"Reading file {file}")
        data.extend(load_json(os.path.join(args.data_path, file)))
        
    logger.info(f"Data loaded, total length {len(data)}")
    assert torch.cuda.is_available()
    model = surrogate_s3()
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt, strict=True)
    model.to("cuda")
    model.eval()
    total_samples = 0
    total_aae = 0
    total_mae = 0
    pbar = tqdm(total=len(data))
    with torch.no_grad():
        for item in data:
            gt = torch.tensor(item["condition"], device="cuda", dtype=torch.float32)
            gen_structure = torch.tensor(item["generation"], device="cuda", dtype=torch.float32)
            # binarization
            mask = gen_structure[0] < torch.mean(gen_structure[0])
            gen_structure[0][mask] = 0
            gen_structure[0][~mask] = torch.max(gen_structure[0]).clip(0, 1).round(decimals=2)
            mask = gen_structure[1] < torch.mean(gen_structure[1])
            gen_structure[1][mask] = 0
            gen_structure[1][~mask] = torch.max(gen_structure[1]).clip(0, 1).round(decimals=2)
            
            # restore to the original size
            full_structure = torch.zeros(3, 2*gen_structure.shape[1], 2*gen_structure.shape[2], device="cuda")
            left_right = torch.cat([gen_structure[0], gen_structure[0].fliplr()], dim=1)
            full_structure[0] = torch.cat([left_right, left_right.flipud()], dim=0)
            
            left_right = torch.cat([gen_structure[1], gen_structure[1].fliplr()], dim=1)
            full_structure[1] = torch.cat([left_right, left_right.flipud()], dim=0)
            
            full_structure[2] = torch.max(gen_structure[2]).clip(0, 1).round(decimals=2)
            
            predicted_spec = model(inputs=full_structure.unsqueeze(0)).prediction
            
            mae = torch.mean(torch.abs(predicted_spec - gt))
            aae = torch.sum(torch.abs(predicted_spec - gt))
            pbar.write(f"r_index {round(torch.max(full_structure[0]).item(), 2)}, thick {round(torch.max(full_structure[1]).item(), 2)}, l_size {round(torch.max(full_structure[2]).item(), 2)} AAE {aae} MAE {mae}")
            total_aae += aae
            total_mae += mae
            pbar.update(1)
           
           
    total_samples = len(data)
    average_aae = total_aae / total_samples
    average_mae = total_mae / total_samples
    print("Average AAE:", average_aae.item())
    print("Average MAE:", average_mae.item())
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    # parser.add_argument("")
    args = parser.parse_args()
    main(args)
    