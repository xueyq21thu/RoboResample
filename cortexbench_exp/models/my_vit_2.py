from .base_model import BaseModel
from .modules.transformer_modules import SmallViT


class ViT_2(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.embedding == 'vit':
            self.feature_extractor = SmallViT(    # 
                image_size=224, patch_size=16, dim=128, depth=2, heads=2, mlp_dim=256, dropout=0.1,
            )
            if self.cfg.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("ViT model type is wrong! The repo only suits for [\"vit\"].")
        