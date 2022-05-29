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

from model import VisionTransformer
from dataset import ImagenetteDataset


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



def eval(model, dataset, device, labelmap: dict, exp_dir: str):
    preds = []
    gts = []
    for img, label in dataset:
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img)

        pred_label = torch.argmax(pred, dim=1)
        preds.append(pred_label.item())
        gts.append(label.item())

    cm = confusion_matrix(gts, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[v for v in labelmap.values()],
    )
    plt.figure(figsize=(14,16))
    disp.plot()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "confusion_matrix.jpg"))
    
    # classification report 
    cls_report = classification_report(
        gts, 
        preds, 
        target_names=[v for v in labelmap.values()], 
        output_dict=True,
    )
    print(cls_report)
    print(type(cls_report))
    
    with open(os.path.join(exp_dir, "classification-report.json"), 'w') as fp: 
        json.dump(cls_report, fp, indent=4)
    
    
    

def main():
    root = "/home/johann/sonstiges/vision-transformer-pytorch"
    data = os.path.join(root, "data")
    imagenette = os.path.join(data, "imagenette2-320")
    annos = os.path.join(imagenette, "noisy_imagenette.csv")

    exp_dir = os.path.join(root, "exps/2022-05-28")
    checkpoint = os.path.join(exp_dir, "checkpoints/final.pth")
    sample_imgs_dir = os.path.join(exp_dir, "sample_images")
    if os.path.isdir(sample_imgs_dir):
        shutil.rmtree(sample_imgs_dir)
    os.makedirs(sample_imgs_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(checkpoint)
    model.to(device)
    model.eval()

    label2id = {
        f: i for i, f in enumerate(list(sorted(os.listdir(
            os.path.join(imagenette, 'train')
        ))))
    }

    id2label = {
        v: k for k, v in label2id.items()
    }
    
    labelmap = {k: label2label[v] for k,v in id2label.items()}

    input_dim = int(32**2 * 3)
    max_seq_len = int((320 / 32) * (320 / 32))


    val_dataset = ImagenetteDataset(
        root=imagenette,
        annos=annos,
        mode='val',
        label2id=label2id,
        img_size=(320,320),
        patch_size=32,
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
        prediction = label2label[id2label[pred_label.item()]]
        gt = label2label[id2label[label.item()]]
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
        
    eval(model, val_dataset, device, labelmap, exp_dir)
    


if __name__ == "__main__":
    main()
