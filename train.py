import os
import torch
from dataset import ImagenetteDataset, get_transform



def main():
    root = "/home/johann/sonstiges/vision-transformer-pytorch"
    data = os.path.join(root, "data")
    imagenette = os.path.join(data, "imagenette2-320")
    annos = os.path.join(imagenette, "noisy_imagenette.csv")
    img_size = (320, 320)
    label2id = {
        f: i for i, f in enumerate(list(sorted(os.listdir(
            os.path.join(imagenette, 'train')
        ))))
    }

    print(label2id)

    train_dataset = ImagenetteDataset(
        root=imagenette,
        annos=annos,
        mode='train',
        label2id=label2id,
        img_size=img_size,
    )

    print(len(train_dataset))

    for img, label in train_dataset:
        print(img.size())
        print(label)
        break



if __name__ == "__main__":
    main()
