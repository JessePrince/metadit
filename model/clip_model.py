import torch
import os
import numpy as np
from torch import nn
import torch.nn.functional as F

from model.spec_encoder import SpectrumEncoderV2, SpectrumDecoder
from timm.models.vision_transformer import VisionTransformer
import torch.distributed as dist

local_rank = int(os.getenv("LOCAL_RANK", -1))


def gather_features(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    local_loss: bool=False,
    gather_with_grad: bool=False,
    rank: int=0,
    world_size: int=1
) -> tuple[torch.Tensor, torch.Tensor]:
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss: bool=False,
        gather_with_grad: bool=False,
        cache_labels: bool=False,
        rank: int=0,
        world_size: int=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                # note how they calculate loss here, they use local left side matrix to multiply the
                # global right side matrix, that's why we must shift the label according to rank
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        # print("logits per image", logits_per_image)
        # print("logits per text", logits_per_text)
        
        # exit(0)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CLIPModel(nn.Module):
    def __init__(
        self,
        ve_output_channel,
        input_size
    ):
        super().__init__()
        self.context_encoder = SpectrumEncoderV2(vae_channel=ve_output_channel*2, num_heads=1)
        self.context_decoder = SpectrumDecoder(num_blocks=[1, 1], num_heads=1, vae_channel=ve_output_channel)
        self.input_size = input_size
        self.ve_output_channel = ve_output_channel
        
        # call forward_features only
        self.vit = VisionTransformer(img_size=input_size, patch_size=2, in_chans=3, embed_dim=384, depth=6, num_heads=6)
        del self.vit.head
        del self.vit.head_drop
        del self.vit.fc_norm
        lshape = []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * np.log(1 / 0.07))
        # self.logit_scale = nn.Parameter(torch.ones(lshape), requires_grad=False)
        self.clip_loss = ClipLoss(rank=local_rank, world_size=4)
        
        self.img_proj = nn.Linear(384, 512, bias=False)
        self.context_proj = nn.Linear(ve_output_channel*301, 512, bias=False)
        
        self.apply(self._init_param)
        
    def _init_param(self, module):
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterize the mu and logvar from VAE to latent

        Args:
            mu (torch.Tensor): model predicted mu
            logvar (torch.Tensor): model predicted logvar

        Returns:
            torch.Tensor: eps * std + mu
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    def _text_encoder(self, condition_context):
        # transformer-like VAE encoder for text
        output = self.context_encoder(condition_context)
        mu, log_var = torch.chunk(output, 2, dim=-1)
             
        z = self._reparameterize(mu, log_var)

        return [z, mu, log_var]
        
    def _img_clip(self, image_input):
        """Get image CLIP latent and logit scale"""
        image_latent = self.vit.forward_features(image_input)
        image_latent = image_latent[:, 0, :] # clstoken

        return image_latent, self.logit_scale
    
    def forward(self, inputs, condition, **kwargs):
        x0, mu, log_var = self._text_encoder(condition)
        recon = self.context_decoder(x0)
        text_feat = self.context_proj(x0.flatten(1))
        
        recon_loss = F.l1_loss(recon, condition)
        
        recon_gt_clip, logit_scale = self._img_clip(inputs)
        recon_gt_clip = self.img_proj(recon_gt_clip)
        image_features = recon_gt_clip.float() / recon_gt_clip.float().norm(dim=-1, keepdim=True)
        text_features = text_feat.float() / text_feat.float().norm(dim=-1, keepdim=True)
        clip_loss = self.clip_loss(image_features.float(), text_features.float(), logit_scale.float())
        
        kld_loss = -0.5 * torch.sum(1 + log_var - (0.3 * mu) ** 6 - log_var.exp(), dim = 1) # KL loss for text VAE
        # kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        kld_loss_weight = 1e-3 # 0.0005
        
        # loss_mlp = recon_loss + clip_loss
        # loss_mlp = clip_loss
        loss_mlp = recon_loss + clip_loss + kld_loss * kld_loss_weight
        # loss_mlp = recon_loss + kld_loss * kld_loss_weight
        # loss_mlp = clip_loss + kld_loss * kld_loss_weight
        
        output = {
            "loss": loss_mlp.mean(),
            "clip_loss": clip_loss.detach().mean(),
            "recon_loss": recon_loss.detach().mean(),
            "kld_loss": kld_loss.detach().mean(),
            "kld_loss_weight": torch.tensor(kld_loss_weight, device=kld_loss.device),
            "loss_clip_logit_scale": logit_scale,
        }
        
        return output
    