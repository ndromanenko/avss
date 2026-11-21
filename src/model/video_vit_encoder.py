import torch
from torch import nn


class VideoViTEncoder(nn.Module):
    def __init__(self, features, image_size=96, patch_size=8, hidden_size=128, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()

        self.patch_embedding = nn.Conv2d(1, hidden_size, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.normal_(self.position_embedding, std=0.02)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ))
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, features)

    def forward(self, video):
        batch_size, speakers, frames, height, width = video.shape
        x = video.reshape(batch_size * speakers * frames, 1, height, width).float() / 255.0
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        x = self.projection(x)
        x = x.reshape(batch_size * speakers, frames, -1).transpose(1, 2)
        return x.reshape(batch_size, -1, frames)
