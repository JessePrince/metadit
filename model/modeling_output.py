from dataclasses import dataclass, asdict
import torch
from collections.abc import Mapping
# from transformers.modeling_outputs
from transformers.utils.generic import ModelOutput

@dataclass
class VAEReturn(ModelOutput):
    loss: torch.Tensor = None
    kl_loss: torch.Tensor = None
    recon_loss: torch.Tensor = None
    z: torch.Tensor = None
    reconstruction: torch.Tensor = None
    
    
@dataclass
class SurrogateOutput(ModelOutput):
    loss: torch.Tensor = None
    prediction: torch.Tensor = None
    
    