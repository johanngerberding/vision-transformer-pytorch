import argparse
import os
import tqdm
import datetime
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ImagenetteDataset, get_transform
from model import VisionTransformer
from utils import get_number_params
from config import get_cfg_defaults 

import wandb


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    num_corrects = 0

    for img, label in tqdm.tqdm(dataloader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(img)

        pred_max = torch.argmax(pred, dim=1)
        correct = torch.sum(label == pred_max) / img.shape[0]
        num_corrects += correct

        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print("Train Loss: {:.5f}".format(epoch_loss / len(dataloader)))
    print("Train Accuracy: {:.2f} %".format(num_corrects / len(dataloader) * 100))

    epoch_loss /= len(dataloader)
    epoch_acc = num_corrects / len(dataloader)

    return epoch_loss, epoch_acc



def val_epoch(model, dataloader, loss_fn, device):
    model.eval()
    epoch_loss = 0
    num_corrects = 0

    for img, label in tqdm.tqdm(dataloader):
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model(img)

        pred_max = torch.argmax(pred, dim=1)
        correct = torch.sum(label == pred_max) / img.shape[0]
        num_corrects += correct

        loss = loss_fn(pred, label)

        epoch_loss += loss.item()

    print("Val Loss: {:.4f}".format(epoch_loss / len(dataloader)))
    print("Val Accuracy: {:.2f} %".format(num_corrects / len(dataloader) * 100))

    epoch_loss /= len(dataloader)
    epoch_acc = num_corrects / len(dataloader)

    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config.yaml different from default config.py", type=str)
    parser.add_argument("--gpu", help="Use gpu or cpu, default is True.", type=bool, default=True)
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.freeze()
    print(cfg)
    
    wandb.init(
        project="vision-transformer-pytorch",
        entity="johanngerber",
        config = {
            "learning_rate": cfg.TRAIN.BASE_LR,
            "epochs": cfg.TRAIN.N_EPOCHS,
            "train_batch_size": cfg.TRAIN.BATCH_SIZE,
            "val_batch_size": cfg.VAL.BATCH_SIZE,
            "image_size": cfg.DATA.IMG_SIZE,
            "patch_size": cfg.DATA.PATCH_SIZE,
            "num_layers": cfg.MODEL.NUM_LAYERS,
            "betas": cfg.OPTIM.BETAS,
            "eps": cfg.OPTIM.EPS,
        },
    )
    config = wandb.config

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    exp_dir = os.path.join(cfg.PROJECT.ROOT, "exps/{}".format(today))
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: 
        device = torch.device('cpu')
    print(f"Device: {device}")

    train_dataset = ImagenetteDataset(
        config=cfg,
        mode='train',
    )

    print(f"Training samples: {len(train_dataset)}")

    val_dataset = ImagenetteDataset(
        config=cfg,
        mode='val',
    )

    print(f"Validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
    )

    print(f"Length train dataloader: {len(train_dataloader)}")

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
    )

    model = VisionTransformer(cfg)
    model.to(device)
    model.train()

    num_params = get_number_params(model)
    print(f"Model has {num_params} parameters.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )

    loss_fn = nn.CrossEntropyLoss().to(device)


    for i in range(config.epochs):
        print(f"Epoch {i+1}")
        train_loss, train_acc = train_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            device,
        )
        val_loss, val_acc = val_epoch(
            model,
            val_dataloader,
            loss_fn,
            device,
        )
        wandb.log(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    out_model_path = os.path.join(checkpoints_dir, "final.pth")
    torch.save(model, out_model_path)
    print(f"Saved model checkpoint: {out_model_path}")



if __name__ == "__main__":
    main()
