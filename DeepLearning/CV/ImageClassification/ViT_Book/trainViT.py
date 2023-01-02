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
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt
from cycler import cycler
from vit import Vit

DATASET = "cifar10"
# DATASET = "imagenet"
if DATASET == "cifar10":
    IMG_SIZE = 32
    BATCH_SIZE = 256
    NUM_CLASSES = 10
elif DATASET == "imagenet":
    IMG_SIZE = 128
    BATCH_SIZE = 128
    NUM_CLASSES = 1000

EPOCHS = 5
NUM_WORKERS = 1

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
        train_dataset = ImageNetDataset(
            annotations_file="../../../../datasets/ImageNet/LOC_train_solution.csv",
            img_dir="../../../../datasets/ImageNet/ILSVRC/Data/CLS-LOC/train",
            transform=train_transform,
            mode="train",
        )

        val_dataset = ImageNetDataset(
            annotations_file="../../../../datasets/ImageNet/LOC_val_solution.csv",
            img_dir="../../../../datasets/ImageNet/ILSVRC/Data/CLS-LOC/val",
            transform=val_transform,
            mode="val",
        )
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            "../../../../datasets/cifar10", transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR10(
            "../../../../datasets/cifar10",
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


def setModel():
    num_classes = NUM_CLASSES
    channel = 3
    model = Vit(in_channels=channel, num_classes=num_classes, image_size=IMG_SIZE)
    return model


def setUtils(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer


def train(epochs, model, criterion, optimizer, train_loader, val_loader, device):
    print(
        "\nTrain data: {} | Validation data: {}".format(
            len(train_loader.dataset), len(val_loader.dataset)
        )
    )
    print("Input shape = {}\n".format(next(iter(train_loader))[0].shape))
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
            train_loss / train_total,
            val_loss / val_total,
            train_acc / train_total,
            val_acc / val_total,
        ]
        results.append(result)
        print(
            "[{}/{}] train loss {:.4f} val loss {:.4f} | train acc {:.4f} val acc {:.4f}".format(
                epoch + 1, epochs, result[1], result[2], result[3], result[4]
            )
        )

    print("Finished Training")
    df = pd.DataFrame(
        data=results,
        columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"],
    )
    saveResults(df)


def saveResults(df):
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m%d-%H%M%S")
    df.to_csv(f"results/result_{now}.csv", index=False)
    plt.figure(figsize=(6, 4), tight_layout=True)
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xticks(df["epoch"])
    plt.legend()
    plt.savefig("results/loss_{now}.jpg")

    plt.figure(figsize=(6, 4), tight_layout=True)
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xticks(df["epoch"])
    plt.legend()
    plt.savefig("results/acc_{now}.jpg")


if __name__ == "__main__":
    device = checkEnv()
    train_transform, val_transform = setTransforms(dataset=DATASET)
    train_dataset, val_dataset = setDatasets(
        train_transform, val_transform, dataset=DATASET
    )
    train_loader, val_loader = setDataloaders(train_dataset, val_dataset)
    model = setModel()
    criterion, optimizer = setUtils(model)
    train(EPOCHS, model, criterion, optimizer, train_loader, val_loader, device)
