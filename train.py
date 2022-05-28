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
    wandb.init(
        project="vision-transformer-pytorch",
        entity="johanngerber",
        config = {
            "learning_rate": 0.0001,
            "epochs": 100,
            "train_batch_size": 256,
            "val_batch_size": 128,
            "image_size": (320,320),
            "patch_size": 32,
            "num_layers": 2,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
        },
    )
    config = wandb.config

    """
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {

        }
    }
    """

    root = "/home/johann/sonstiges/vision-transformer-pytorch"
    data = os.path.join(root, "data")
    imagenette = os.path.join(data, "imagenette2-320")
    annos = os.path.join(imagenette, "noisy_imagenette.csv")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    exp_dir = os.path.join(root, "exps/{}".format(today))
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    label2id = {
        f: i for i, f in enumerate(list(sorted(os.listdir(
            os.path.join(imagenette, 'train')
        ))))
    }

    input_dim = int(config.patch_size**2 * 3)
    max_seq_len = int(
        (config.image_size[0] / config.patch_size) *
        (config.image_size[1] / config.patch_size))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(label2id)

    train_dataset = ImagenetteDataset(
        root=imagenette,
        annos=annos,
        mode='train',
        label2id=label2id,
        img_size=config.image_size,
        patch_size=config.patch_size,
    )

    print(f"Training samples: {len(train_dataset)}")

    val_dataset = ImagenetteDataset(
        root=imagenette,
        annos=annos,
        mode='val',
        label2id=label2id,
        img_size=config.image_size,
        patch_size=config.patch_size,
    )

    print(f"Validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=8,
    )

    print(f"Length train dataloader: {len(train_dataloader)}")

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = VisionTransformer(
        max_seq_len=max_seq_len,
        num_layers=config.num_layers,
        input_dim=input_dim,
    )
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
