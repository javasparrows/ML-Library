import os
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms as T
from vit import Vit

IMG_SIZE = 256
EPOCHS = 5

print(torch.__version__)
print(torchvision.__version__)
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"DEVICE = {DEVICE}")


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


def setDataloaders():
    train_transform = T.Compose(
        [
            T.ToTensor(),
            T.RandomResizedCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageNetDataset(
        annotations_file="../../../../datasets/ImageNet/LOC_train_solution.csv",
        img_dir="../../../../datasets/ImageNet/ILSVRC/Data/CLS-LOC/train",
        transform=train_transform,
        mode="train",
    )

    valid_transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize([IMG_SIZE, IMG_SIZE]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_dataset = ImageNetDataset(
        annotations_file="../../../../datasets/ImageNet/LOC_val_solution.csv",
        img_dir="../../../../datasets/ImageNet/ILSVRC/Data/CLS-LOC/val",
        transform=valid_transform,
        mode="val",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True
    )
    return train_dataset, val_dataset, train_loader, val_loader


def setModel(train_dataset):
    num_classes = len(np.unique(train_dataset.getLabels()[0]))  # 1000
    channel = 3
    model = Vit(in_channels=channel, num_classes=num_classes, image_size=IMG_SIZE)
    return model.to(DEVICE)


def setUtils(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer


def train(epochs, criterion, optimizer, train_loader, val_loader):
    results = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = 0
        train_acc = 0
        train_total = 0
        for data in tqdm(train_loader):
            model.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

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

        # if i % printEpochs == (printEpochs - 1):    # print every 10 mini-batches
        val_loss = 0
        val_acc = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader):
                inputs, labels = data
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_acc += (outputs.argmax(axis=1) == labels).sum().item()
                val_total += len(labels)

        results.append(
            [
                epoch + 1,
                train_loss / train_total,
                val_loss / val_total,
                train_acc / train_total,
                val_acc / val_total,
            ]
        )
        print(
            "[{}/{}] train loss {:.4f} val loss {:.4f} | train acc {:.4f} val acc {:.4f}".format(
                epoch + 1, epochs, results[1], results[2], results[3], results[4]
            )
        )

    print("Finished Training")
    df = pd.DataFrame(
        data=results,
        columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"],
    )
    df.to_csv("ViT_result.csv")


if __name__ == "__main__":
    train_dataset, val_dataset, train_loader, val_loader = setDataloaders()
    model = setModel(train_dataset)
    criterion, optimizer = setUtils(model)
    train(EPOCHS, criterion, optimizer, train_loader, val_loader)
