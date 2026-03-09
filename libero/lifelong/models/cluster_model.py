import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ClusterModel(nn.Module):
    """
    A learnable clustering model for tasks.
    - Encodes task_emb to latent (hyper_dim).
    - Learns centroids for clusters.
    - Assigns cluster_id by argmin dist.
    - Returns hyper as the assigned centroid.
    """
    def __init__(self, input_dim, num_clusters, hyper_dim, temperature=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hyper_dim),
            nn.LayerNorm(hyper_dim),          # 关键：稳定训练
        )
        self.centroids = nn.Parameter(torch.randn(num_clusters, hyper_dim) * 0.1)
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.hyper_dim = hyper_dim

    def forward(self, task_emb):
        latent = self.encoder(task_emb)                    # (B, hyper_dim)
        latent = F.normalize(latent, dim=-1)               # 单位球
        centroids_norm = F.normalize(self.centroids, dim=-1)
        dists = torch.cdist(latent, centroids_norm)        # (B, K)

        cluster_id = torch.argmin(dists, dim=-1)           # hard assignment
        hyper = self.centroids[cluster_id]                 # (B, hyper_dim)

        return cluster_id, hyper, latent, dists