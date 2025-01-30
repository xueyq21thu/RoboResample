import torch

from .base_trifinger import BaseAlgo_Trifinger
from ..utils.mmd_utils import MMD_loss
from ..utils.kl_div_utils import KL_div_loss_with_knn


class BC_Policy_Trifinger(BaseAlgo_Trifinger):
    def __init__(self, cfg, inference=False):
        super().__init__(cfg, inference)
        self.mmd = MMD_loss()
        self.kl_div = KL_div_loss_with_knn()

    def forward_backward(self, data):
        bc_loss, mmd, kl_div = self.compute_loss(data)
        loss = bc_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.grad_clip)
        self.optimizer.step()

        ret_dict = {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            "mmd": mmd.item(),
            "kl_div": kl_div.item(),
        }

        return ret_dict
    
    def compute_loss(self, data, return_meric=True):
        x, z, dist = self.model(data, return_latent=True)
        bc_loss = self.model.policy_head.loss_fn(dist, data["output"]["action"].to(self.device), reduction="mean")

        if return_meric:
            with torch.no_grad():
                mmd = self.mmd(x, z)
                kl_div = self.kl_div(x, z, k=5)

            return bc_loss, mmd, kl_div

        return bc_loss

