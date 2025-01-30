""" KL-Divergence estimation through K-Nearest Neighbours

    This module provides four implementations of the K-NN divergence estimator of
        Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú.
        "Divergence estimation for multidimensional densities via
        k-nearest-neighbor distances." Information Theory, IEEE Transactions on
        55.5 (2009): 2392-2405.

    The implementations are through:
        numpy (naive_estimator)
        scipy (scipy_estimator)
        scikit-learn (skl_estimator / skl_efficient)

    No guarantees are made w.r.t the efficiency of these implementations.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KL_div_loss_with_knn(nn.Module):
    def __init__(self):
        super().__init__()

    def knn_distance(self, point, sample, k):
        """Euclidean distance from `point` to it's `k`-Nearest
        Neighbour in `sample`

        This function works for points in arbitrary dimensional spaces.
        """
        # Compute all euclidean distances
        norms = torch.norm(sample - point, dim=1)
        # Return the k-th nearest
        if len(norms) > k:
            return torch.sort(norms)[0][k]
        else:
            return torch.sort(norms)[0][len(norms)-1]

    def verify_sample_shapes(self, s1, s2):
        # Expects [N, D]
        assert len(s1.shape) == len(s2.shape) == 2
        # Check dimensionality of sample is identical
        assert s1.shape[1] == s2.shape[1]

    def naive_estimator(self, s1, s2, k=1):
        """KL-Divergence estimator using brute-force (torch) k-NN
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
        """
        s1 = s1.reshape(s1.shape[0], -1)
        s2 = s2.reshape(s2.shape[0], -1)
        self.verify_sample_shapes(s1, s2)

        n, m = len(s1), len(s2)
        D = torch.log(torch.tensor(m / (n - 1))).to(s1.device)
        d = float(s1.shape[1])

        for p1 in s1:
            nu = self.knn_distance(p1, s2, k - 1)  # -1 because 'p1' is not in 's2'
            rho = self.knn_distance(p1, s1, k)
            if rho == 0:
                rho = 1e-10
                # raise ValueError('rho must not be 0!')
            D += (d / n) * torch.log(nu / rho)
        return D

    @torch.no_grad()
    def forward(self, s1, s2, k=1):
        return self.naive_estimator(s1, s2, k=k)


class KL_div_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_kl_divergence(self, p, q):
        p = p.reshape(p.shape[0], -1)
        q = q.reshape(q.shape[0], -1)
        p = F.softmax(p, dim=-1)  
        q = F.softmax(q, dim=-1)  
        kl_div = torch.sum(p * torch.log(p / (q + 1e-8)), dim=-1)
        return kl_div.mean()

    @torch.no_grad()
    def forward(self, p, q):
        return self.compute_kl_divergence(p, q)

        