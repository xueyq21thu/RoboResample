import os
from vc_models.models.vit.model_utils import load_model as load_vc1

from .base_model import BaseModel


class VC1(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.embedding == 'vc1_vitb' or 'vc1_vitl':
            self.feature_extractor, _, _, _ = load_vc1(cfg.policy.embedding, load_path=os.path.join(cfg.policy.embedding_dir, 'vc-1'))
            if self.cfg.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("VC-1 model type is wrong! The repo only suits for [\"vc1_vitb\"].")
        