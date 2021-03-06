import os
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def preprocess(img, patch_size: int):
    "Create patches from image tensor"
    patches = []
    assert img.size(1) % patch_size == 0
    assert img.size(2) % patch_size == 0

    for i in range(img.size(1) // patch_size):
        for j in range(img.size(2) // patch_size):
            patch = img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patch = torch.flatten(patch)
            patches.append(patch)

    return patches


class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: str,
        mode: str,
    ):
        self.root = config.DATA.ROOT
        self.mode = mode
        self.imgs = self.load_annos(config.DATA.ANNOS, self.mode)
        self.transform = get_transform(self.mode, config.DATA.IMG_SIZE)
        self.label2id = {l: i for i,l in enumerate(config.DATA.LABEL2ID)}
        self.patch_size = config.DATA.PATCH_SIZE


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        assert os.path.isfile(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        folder = img_path.split('/')[-2]
        label = self.label2id[folder]
        label = torch.tensor(label, dtype=torch.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        patches = preprocess(img, self.patch_size)
        patches = torch.stack(patches)

        return patches, label


    def __len__(self):
        return len(self.imgs)


    def load_annos(self, path: str, mode: str):
        annos = pd.read_csv(path)
        imgs = annos.loc[:, 'path']
        imgs = [img for img in imgs if mode in img]
        return imgs


    def get_img_path(self, idx: int):
        return os.path.join(self.root, self.imgs[idx])


def get_transform(mode: str, img_size: tuple):
    if mode == 'train':
        return A.Compose([
            A.RandomCrop(320, 320, always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.4),
            A.GaussianBlur(p=0.3),
            A.RGBShift(p=0.3),
            A.Rotate(p=0.3),
            A.ChannelDropout(p=0.3),
            A.Cutout(p=0.3),
            A.Resize(img_size[0], img_size[1], p=1.0),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=True,
                p=1.0,
            ),
            ToTensorV2(),
        ])

    elif mode == 'val':
        return A.Compose([
            A.RandomCrop(320, 320, always_apply=True, p=1.0),
            A.Resize(img_size[0], img_size[1], p=1.0),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=True,
                p=1.0,
            ),
            ToTensorV2(),
        ])

    else:
        raise ValueError




