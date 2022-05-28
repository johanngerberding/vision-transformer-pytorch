import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import preprocess


class VisionTransformer(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        num_layers: int,
        input_dim: int,
        d_model: int = 1024,
        n_head: int = 8,
        dim_feedforward: int = 2048,
        n_classes: int = 10,
        pdrop=0.1,
    ):
        super(VisionTransformer, self).__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=pdrop,
            activation=F.gelu,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers)

        self.mlp = nn.Linear(d_model, n_classes)
        self.drop = nn.Dropout(p=pdrop)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, x):
        x = self.embed(x)

        # class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # add positional embedding
        x = x + self.pos

        x = self.drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        cls_token_head = x[:, 0]

        x = self.mlp(cls_token_head)
        return x

