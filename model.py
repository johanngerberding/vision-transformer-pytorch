import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        self.embed = nn.Linear(config.MODEL.INPUT_DIM, config.MODEL.D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, config.MODEL.MAX_SEQ_LEN + 1, config.MODEL.D_MODEL))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.MODEL.D_MODEL))

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.MODEL.D_MODEL,
            nhead=config.MODEL.NUM_HEADS,
            dim_feedforward=config.MODEL.D_FF,
            dropout=config.MODEL.DROPOUT,
            activation=F.gelu,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=config.MODEL.NUM_LAYERS)

        self.mlp = nn.Linear(config.MODEL.D_MODEL, config.DATA.NUM_CLASSES)
        self.drop = nn.Dropout(p=config.MODEL.DROPOUT)
        self.norm = nn.LayerNorm(config.MODEL.D_MODEL, eps=1e-6)


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

