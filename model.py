import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import preprocess


class VisionTransformer(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        d_model: int = 512,
        n_head: int = 8,
        n_classes: int = 10,
        pdrop=0.,
    ):
        super(VisionTransformer, self).__init__()
        self.embed = nn.Linear(768, d_model)
        self.pos = nn.Parameter(torch.zeros(max_seq_len + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, activation=F.gelu)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)

        self.mlp = nn.Linear(d_model, n_classes)
        self.drop = nn.Dropout(p=pdrop)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, x):
        x = self.embed(x)

        # class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos

        x = self.drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        cls_token_head = x[:, 0]

        x = self.mlp(cls_token_head)
        return x



def main():
    test = torch.randn(3, 160, 160)
    patch_size = 16
    patches = preprocess(test, patch_size)
    patches = torch.stack(patches)
    patches = patches.unsqueeze(0)
    print(patches.size())

    vision_transformer = VisionTransformer(max_seq_len=100, d_model=512)

    out = vision_transformer(patches)
    print(out.size())



if __name__ == "__main__":
    main()
