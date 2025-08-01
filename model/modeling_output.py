from dataclasses import dataclass, asdict
import torch
from collections.abc import Mapping
# from transformers.modeling_outputs
from transformers.utils.generic import ModelOutput

# @dataclass
# class BaseReturn:
#     def __getitem__(self, key):
#         return getattr(self, key)

#     def __iter__(self):
#         return iter(asdict(self))

#     def __len__(self):
#         return len(asdict(self))

#     def items(self):
#         return asdict(self).items()

#     def keys(self):
#         return asdict(self).keys()

#     def values(self):
#         return asdict(self).values()


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
    
    