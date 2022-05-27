import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ImagenetteDataset, get_transform
from model import VisionTransformer


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    for img, label in tqdm.tqdm(dataloader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(img)

        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print("Train Loss: {:.4f}".format(epoch_loss / len(dataloader)))


def val_epoch(model, dataloader, loss_fn, device):
    model.eval()
    epoch_loss = 0
    for img, label in tqdm.tqdm(dataloader):
        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = model(img)

        loss = loss_fn(pred, label)

        epoch_loss += loss.item()

    print("Val Loss: {:.4f}".format(epoch_loss / len(dataloader)))




def main():
    root = "/home/johann/sonstiges/vision-transformer-pytorch"
    data = os.path.join(root, "data")
    imagenette = os.path.join(data, "imagenette2-320")
    annos = os.path.join(imagenette, "noisy_imagenette.csv")
    img_size = (160, 160)
    label2id = {
        f: i for i, f in enumerate(list(sorted(os.listdir(
            os.path.join(imagenette, 'train')
        ))))
    }
    patch_size = 16
    lr = 0.003
    betas = (0.9, 0.999)
    eps = 1e-08

    n_epochs = 25
    train_batch_size = 64
    val_batch_size = 32

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

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = VisionTransformer(
        max_seq_len=100
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
    )

    loss_fn = nn.CrossEntropyLoss().to(device)


    for i in range(n_epochs):
        print(f"Epoch {i+1}")
        train_epoch(model, train_dataloader, optimizer, loss_fn, device)
        val_epoch(model, val_dataloader, loss_fn, device)



if __name__ == "__main__":
    main()
