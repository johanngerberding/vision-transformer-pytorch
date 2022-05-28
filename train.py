import os
import tqdm
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
    wandb.init(project="vision-transformer-pytorch", entity="johanngerber")
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
    patch_size = 32
    lr = 0.003
    betas = (0.9, 0.999)
    eps = 1e-08

    n_epochs = 50
    train_batch_size = 128
    val_batch_size = 64

    max_seq_len = int((img_size[0] / patch_size) * (img_size[1] / patch_size))
    num_layers = 4
    input_dim = int(patch_size**2 * 3)

    wandb.config = {
        "learning_rate": 0.003,
        "epochs": 50,
        "train_batch_size": 128,
        "val_batch_size": 64,
        "image_size": (320, 320),
        "patch_size": 32,
        "num_layers": 4,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(label2id)

    train_dataset = ImagenetteDataset(
        root=imagenette,
        annos=annos,
        mode='train',
        label2id=label2id,
        img_size=img_size,
        patch_size=patch_size,
    )

    print(f"Training samples: {len(train_dataset)}")

    val_dataset = ImagenetteDataset(
        root=imagenette,
        annos=annos,
        mode='val',
        label2id=label2id,
        img_size=img_size,
        patch_size=patch_size,
    )

    print(f"Validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8,
    )

    print(f"Length train dataloader: {len(train_dataloader)}")

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = VisionTransformer(
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        input_dim=input_dim,
    )
    model.to(device)
    model.train()

    num_params = get_number_params(model)
    print(f"Model has {num_params} parameters.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
    )

    loss_fn = nn.CrossEntropyLoss().to(device)


    for i in range(n_epochs):
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



if __name__ == "__main__":
    main()
