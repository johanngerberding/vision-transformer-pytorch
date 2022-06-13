import argparse
import os
import json
import torch
import tqdm 
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay,
                             classification_report)

import matplotlib.pyplot as plt

from config import get_cfg_defaults
from dataset import ImagenetteDataset


def eval(
    model,
    dataset,
    device,
    labelmap: list,
    exp_dir: str
):
    preds = []
    gts = []
    for img, label in tqdm.tqdm(dataset):
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img)

        pred_label = torch.argmax(pred, dim=1)
        preds.append(pred_label.item())
        gts.append(label.item())

    cm = confusion_matrix(gts, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labelmap,
    )
    plt.figure(figsize=(16,16))
    disp.plot()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "confusion_matrix.jpg"))

    # classification report
    cls_report = classification_report(
        gts,
        preds,
        target_names=labelmap,
        output_dict=True,
    )

    with open(os.path.join(exp_dir, "classification-report.json"), 'w') as fp:
        json.dump(cls_report, fp, indent=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Experiment directory", type=str)
    parser.add_argument("--model", help="Model checkpoint filename", type=str)
    parser.add_argument("--config", help="Path to alternative config.yaml", type=str)
    parser.add_argument("--gpu", help="Use gpu device if possible", default=True)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)

    cfg.freeze()

    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if args.model:
        model = torch.load(args.model)
    else:
        model = torch.load(os.path.join(args.exp_dir, "checkpoints/final.pth"))

    model.to(device)
    model.eval()

    dataset = ImagenetteDataset(config=cfg, mode='val')
    labelmap = cfg.DATA.LABELNAMES

    eval(model, dataset, device, labelmap, args.exp_dir)


if __name__ == "__main__":
    main()
