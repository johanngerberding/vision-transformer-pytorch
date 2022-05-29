import argparse
import os
import torch
import random
import shutil
import cv2
import json 
from sklearn.metrics import (confusion_matrix, 
                             ConfusionMatrixDisplay, 
                             classification_report)
import matplotlib.pyplot as plt

from dataset import ImagenetteDataset
from config import get_cfg_defaults


label2label = {
    'n01440764': 'fish',
    'n02102040': 'dog',
    'n02979186': 'radio',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'french horn',
    'n03417042': 'garbage truck',
    'n03425413': 'tank column',
    'n03445777': 'golf ball',
    'n03888257': 'paraglider',
}

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Path to experiment directory", type=str)
    parser.add_argument("--config", help="Path to alternative config.yaml", type=str)
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    
    if args.config:
        cfg.merge_from_file(args.config)
    elif os.path.isfile(os.path.join(args.exp_dir, "config.yaml")):
        cfg.merge_from_file(os.path.join(args.exp_dir, "config.yaml"))
    
    cfg.freeze()
    print(cfg)

    checkpoint = os.path.join(args.exp_dir, "checkpoints/final.pth")
    sample_imgs_dir = os.path.join(args.exp_dir, "sample_images")
    if os.path.isdir(sample_imgs_dir):
        shutil.rmtree(sample_imgs_dir)
    os.makedirs(sample_imgs_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(checkpoint)
    model.to(device)
    model.eval()

    val_dataset = ImagenetteDataset(
        config=cfg,
        mode='val',
    )

    print(f"Validation samples: {len(val_dataset)}")

    img_idxs = random.sample([i for i in range(len(val_dataset))], 10)

    print(img_idxs)

    for idx in img_idxs:
        img, label = val_dataset[idx]
        img = img.unsqueeze(0).to(device)
        img_path = val_dataset.get_img_path(idx)

        with torch.no_grad():
            pred = model(img)

        pred_label = torch.argmax(pred, dim=1)
        prediction = cfg.DATA.LABELNAMES[pred_label.item()]
        gt = cfg.DATA.LABELNAMES[label.item()]
        print(img_path)
        print(f"Ground truth: {gt}  <> Prediction: {prediction}")

        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(12,8))
        plt.title(f"Image: {os.path.split(img_path)[1]}\nprediction: {prediction}")
        plt.imshow(im)
        plt.axis('off')
        out = os.path.join(sample_imgs_dir, os.path.split(img_path)[1])
        plt.savefig(out)

    
if __name__ == "__main__":
    main()
