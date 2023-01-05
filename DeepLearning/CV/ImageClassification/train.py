import os
import sys
import yaml
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import matplotlib.pyplot as plt
from cycler import cycler
from models.vit import ViT
from models.alexnet import AlexNet


plt.rcParams["axes.prop_cycle"] = cycler(
    color=[
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14E",
        "#EDC949",
        "#B07AA2",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
)


def checkEnv(cfg):
    print(torch.__version__)
    print(torchvision.__version__)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"device = {device}")

    torch.backends.cudnn.benchmark = cfg["BENCHMARK"]
    print(f"cudnn benchmark = {cfg['BENCHMARK']}")
    return device


class ImageNetDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=None,
        target_transform=None,
        mode="train",
    ):
        self.df = pd.read_csv(annotations_file)
        self.labelNums = np.array(self.getLabels()[0]).astype(np.uint8)
        self.labelNames = self.getLabels()[1]
        self.img_names = self.getImgNames()
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

    def __len__(self):
        return len(self.labelNums)

    def __getitem__(self, idx):
        labelNum = self.labelNums[idx]
        labelName = self.labelNames[idx]
        if self.mode == "train":
            img_path = (
                os.path.join(self.img_dir, labelName, self.img_names[idx]) + ".JPEG"
            )
        else:
            img_path = os.path.join(self.img_dir, self.img_names[idx]) + ".JPEG"
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            labelNum = self.target_transform(labelNum)
        return img, labelNum

    def getLabels(self):
        labelNames = [
            row.split(" ")[0] for row in self.df["PredictionString"].values.tolist()
        ]
        le = LabelEncoder()
        labelNums = le.fit_transform(labelNames).tolist()
        lableMap = dict(zip(le.classes_, range(len(le.classes_))))
        return labelNums, labelNames, lableMap

    def getImgNames(self):
        return self.df["ImageId"].values.tolist()


def setTransforms(cfg):
    dataset = cfg["DATASET"]
    img_size = cfg["IMG_SIZE"]
    if dataset == "imagenet":
        train_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_transform = T.Compose(
            [
                T.ToTensor(),
                T.CenterCrop(img_size),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif dataset == "cifar10":
        train_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize([img_size, img_size]),
                T.RandomHorizontalFlip(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize([img_size, img_size]),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return train_transform, val_transform


def setDatasets(train_transform, val_transform, cfg):
    dataset = cfg["DATASET"]
    print(f"Use {dataset} dataset.")
    if dataset == "imagenet":
        rootDir = "../../../datasets/ImageNet/"
        train_dataset = ImageNetDataset(
            annotations_file=os.path.join(rootDir, "LOC_train_solution.csv"),
            img_dir=os.path.join(rootDir, "ILSVRC/Data/CLS-LOC/train"),
            transform=train_transform,
            mode="train",
        )

        val_dataset = ImageNetDataset(
            annotations_file=os.path.join(rootDir, "LOC_val_solution.csv"),
            img_dir=os.path.join(rootDir, "ILSVRC/Data/CLS-LOC/val"),
            transform=val_transform,
            mode="val",
        )
    elif dataset == "cifar10":
        rootDir = "../../../datasets/cifar10"
        train_dataset = datasets.CIFAR10(
            rootDir, transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR10(
            rootDir,
            train=False,
            transform=val_transform,
            download=True,
        )
    return train_dataset, val_dataset


def setDataloaders(train_dataset, val_dataset, cfg):
    batch_size = cfg["BATCH_SIZE"]
    num_workers = cfg["NUM_WORKERS"]
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def setModel(cfg):
    model_name = cfg["MODEL_NAME"]
    num_classes = cfg["NUM_CLASSES"]
    image_size = cfg["IMG_SIZE"]
    if model_name == "AlexNet":
        model = AlexNet(
            num_classes=num_classes,
            dropout=0.1,
        )
    if model_name == "ViT":
        model = ViT(
            in_channels=3,
            num_classes=num_classes,
            num_patch_row=cfg["NUM_PATCH_ROW"],
            image_size=image_size,
            dropout=0.1,
        )
    return model


def setUtils(model, cfg):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=cfg["LR"], momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(
        optimizer,
        step_size=cfg["SCHEDULER"]["STEP_SIZE"],
        gamma=cfg["SCHEDULER"]["GAMMA"],
    )
    return criterion, optimizer, scheduler


def train(
    model, criterion, optimizer, scheduler, train_loader, val_loader, device, cfg
):
    epochs = cfg["EPOCHS"]
    use_amp = cfg["USE_AMP"]
    print(
        "\nTrain data: {} | Validation data: {}".format(
            len(train_loader.dataset), len(val_loader.dataset)
        )
    )
    print(f"Input shape = {next(iter(train_loader))[0].shape}")
    print(f"{epochs} epochs training is going to run.")
    print(f"Automatic Mixed Precision mode is {use_amp}.\n")

    now = datetime.now()
    now = now.strftime("%Y-%m%d-%H%M")

    model = model.to(device)
    results = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = 0
        train_acc = 0
        train_total = 0
        for data in tqdm(train_loader):
            model.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                # forward + backward + optimize
                outputs = model(inputs)
                # print(outputs)
                loss = criterion(outputs, labels)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

            # print statistics
            train_loss += loss.item()
            train_acc += (outputs.argmax(axis=1) == labels).sum().item()
            train_total += len(labels)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_loss = 0
        val_acc = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_acc += (outputs.argmax(axis=1) == labels).sum().item()
                val_total += len(labels)

        result = [
            epoch + 1,
            lr,
            train_loss / len(train_loader),
            val_loss / len(val_loader),
            train_acc / train_total,
            val_acc / val_total,
        ]
        results.append(result)
        print(
            "[{}/{}] lr: {:.6f} train loss {:.4f} val loss {:.4f} | train acc {:.4f} val acc {:.4f}".format(
                epoch + 1, epochs, lr, result[2], result[3], result[4], result[5]
            )
        )
        saveResults(results, now, cfg)

    print("Finished Training")


def saveResults(results, now, cfg):
    dataset = cfg["DATASET"]
    model_name = cfg["MODEL_NAME"]
    saveDir = f"results/{dataset}/{now}_{model_name}"
    os.makedirs(saveDir, exist_ok=True)

    df = pd.DataFrame(
        data=results,
        columns=["epoch", "lr", "train_loss", "val_loss", "train_acc", "val_acc"],
    )

    df.to_csv(os.path.join(saveDir, f"result_{now}.csv"), index=False)
    plt.figure(figsize=(6, 10), tight_layout=True)
    plt.subplot(3, 1, 1)
    plt.title(f"{model_name} {dataset} {now}")
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    # plt.figure(figsize=(6, 4), tight_layout=True)
    plt.title(f"{model_name} {dataset} {now}")
    plt.subplot(3, 1, 2)
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    # plt.figure(figsize=(6, 4), tight_layout=True)
    plt.subplot(3, 1, 3)
    plt.title(f"{model_name} {dataset} {now}")
    plt.plot(df["epoch"], df["lr"], label="lr")
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.legend()
    plt.savefig(os.path.join(saveDir, f"result.jpg"))
    plt.close()


def readCfg(path):
    try:
        with open(path) as file:
            obj = yaml.safe_load(file)
            print(obj)
            return obj
    except Exception as e:
        print("Exception occurred while loading YAML...", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cfg = readCfg("config/vit_cifar10.yaml")
    # cfg = readCfg("config/vit_imagenet.yaml")
    device = checkEnv(cfg)
    train_transform, val_transform = setTransforms(cfg)
    train_dataset, val_dataset = setDatasets(train_transform, val_transform, cfg)
    train_loader, val_loader = setDataloaders(train_dataset, val_dataset, cfg)
    model = setModel(cfg)
    criterion, optimizer, scheduler = setUtils(model, cfg)
    train(model, criterion, optimizer, scheduler, train_loader, val_loader, device, cfg)
