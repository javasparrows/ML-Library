import os
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

# DATASET = "cifar10"
DATASET = "imagenet"
if DATASET == "cifar10":
    IMG_SIZE = 32
    BATCH_SIZE = 2048
    NUM_CLASSES = 10
    LR = 0.01
elif DATASET == "imagenet":
    IMG_SIZE = 224
    BATCH_SIZE = 512
    NUM_CLASSES = 1000
    LR = 0.01

# MODEL_NAME = "SimpleCNN"
MODEL_NAME = "AlexNet"
# MODEL_NAME = "ViT"

EPOCHS = 60
NUM_WORKERS = 2
STEP_SIZE = 20

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


def checkEnv():
    print(torch.__version__)
    print(torchvision.__version__)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"device = {device}")
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


def setTransforms(dataset="cifar10"):
    if dataset == "imagenet":
        train_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(IMG_SIZE),
                T.RandomHorizontalFlip(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize([IMG_SIZE, IMG_SIZE]),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif dataset == "cifar10":
        train_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return train_transform, val_transform


def setDatasets(train_transform, val_transform, dataset="cifar10"):
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


def setDataloaders(train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def setModel(model_name):
    num_classes = NUM_CLASSES
    if model_name == "AlexNet":
        model = AlexNet(
            num_classes=num_classes,
            dropout=0.1,
        )
    if model_name == "ViT":
        model = ViT(
            in_channels=3,
            num_classes=num_classes,
            num_patch_row=7,
            image_size=IMG_SIZE,
            dropout=0.1,
        )
    return model


def setUtils(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LR, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=0.5)
    return criterion, optimizer, scheduler


def train(
    epochs,
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    model_name,
):
    print(
        "\nTrain data: {} | Validation data: {}".format(
            len(train_loader.dataset), len(val_loader.dataset)
        )
    )
    print(f"Input shape = {next(iter(train_loader))[0].shape}")
    print(f"{epochs} epochs training is going to run.\n")

    now = datetime.now()
    now = now.strftime("%Y-%m%d-%H%M")

    model = model.to(device)
    results = []
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

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
            train_loss / train_total,
            val_loss / val_total,
            train_acc / train_total,
            val_acc / val_total,
        ]
        results.append(result)
        print(
            "[{}/{}] lr: {:.6f} train loss {:.4f} val loss {:.4f} | train acc {:.4f} val acc {:.4f}".format(
                epoch + 1, epochs, lr, result[2], result[3], result[4], result[5]
            )
        )
        saveResults(results, now, model_name)

    print("Finished Training")


def saveResults(results, now, model_name):
    saveDir = f"results/{DATASET}/{now}_{model_name}"
    os.makedirs(saveDir, exist_ok=True)

    df = pd.DataFrame(
        data=results,
        columns=["epoch", "lr", "train_loss", "val_loss", "train_acc", "val_acc"],
    )

    df.to_csv(os.path.join(saveDir, f"result_{now}.csv"), index=False)
    plt.figure(figsize=(6, 4), tight_layout=True)
    plt.title(f"{model_name} {DATASET} {now}")
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xticks(df["epoch"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(saveDir, f"loss.jpg"))
    plt.close()

    plt.figure(figsize=(6, 4), tight_layout=True)
    plt.title(f"{model_name} {DATASET} {now}")
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xticks(df["epoch"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join(saveDir, f"acc.jpg"))
    plt.close()

    plt.figure(figsize=(6, 4), tight_layout=True)
    plt.title(f"{model_name} {DATASET} {now}")
    plt.plot(df["epoch"], df["lr"], label="lr")
    plt.xticks(df["epoch"])
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.legend()
    plt.savefig(os.path.join(saveDir, f"lr.jpg"))
    plt.close()


if __name__ == "__main__":
    device = checkEnv()
    train_transform, val_transform = setTransforms(dataset=DATASET)
    train_dataset, val_dataset = setDatasets(
        train_transform, val_transform, dataset=DATASET
    )
    train_loader, val_loader = setDataloaders(train_dataset, val_dataset)
    model = setModel(MODEL_NAME)
    criterion, optimizer, scheduler = setUtils(model)
    train(
        EPOCHS,
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        MODEL_NAME,
    )
