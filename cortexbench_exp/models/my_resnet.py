import torch
import numpy as np
from einops import rearrange

from .base_model import BaseModel
from .modules.image_encoder import ResnetEncoder


class ResNet(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        cfg.policy.embedding_dim = 512
        if cfg.policy.embedding == 'resnet18':
            self.feature_extractor = ResnetEncoder(output_size=cfg.policy.embedding_dim)    # 0.978752 M
            if self.cfg.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("ResNet model type is wrong! The repo only suits for [\"resnet18\"].")
        