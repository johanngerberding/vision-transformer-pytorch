import os
import torch

import matplotlib.pyplot as plt

from model import VisionTransformer
from dataset import ImagenetteDataset



def main():
    root = "/home/johann/sonstiges/vision-transformer-pytorch"
    data = os.path.join(root, "data")
    imagenette = os.path.join(data, "imagenette2-320")
    annos = os.path.join(imagenette, "noisy_imagenette.csv")
    checkpoint = os.path.join(root, "exps/2022-05-28/checkpoints/final.pth")

    model = torch.load(checkpoint)
    model.eval()

    label2id = {
        f: i for i, f in enumerate(list(sorted(os.listdir(
            os.path.join(imagenette, 'train')
        ))))
    }

    input_dim = int(32**2 * 3)
    max_seq_len = int((320 / 32) * (320 / 32))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_dataset = ImagenetteDataset(
        root=imagenette,
        annos=annos,
        mode='val',
        label2id=label2id,
        img_size=(320,320),
        patch_size=32,
    )

    print(f"Validation samples: {len(val_dataset)}")

    img_idxs = random.sample([i for i in range(len(val_dataset))], 5)

    print(img_idxs)

    for idx in img_idxs:
        img, label = val_dataset[idx]
        print(img.size())
        print(label)



if __name__ == "__main__":
    main()
