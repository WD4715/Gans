import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from albumentations import (
    Compose, HorizontalFlip,
    RandomGamma,
    ToFloat, ShiftScaleRotate
)
import albumentations
class CustomDataset(Dataset):
    def __init__(self, train, train_path, test_path, transform = None):
        super(CustomDataset, self).__init__()

        self.train = train

        train_csv_file = pd.read_csv(train_path)
        train_img_path = train_csv_file['img_path']
        self.train_img_path = train_img_path.values.tolist()
        self.train_img_path = np.array(self.train_img_path)

        test_csv_file = pd.read_csv(test_path)
        test_csv_path = test_csv_file['img_path']
        test_img_path = test_csv_path.values.tolist()
        self.test_img_path = np.array(test_img_path)
        self.transform = transform

    def __getitem__(self, idx):

        if self.train:
            img_path = self.train_img_path[idx]
            img = np.load(img_path)
            img = (img - img.min()) / (img.max() - img.min())
            if self.transform != None:
                img = self.transform(image = img)['image']

            img = torch.tensor(img, dtype = torch.float32)
            img = img.permute(2, 0, 1)
            return img

        else:
            img_path = self.test_img_path[idx]
            img = np.load(img_path)
            img = (img - img.min()) / (img.max() - img.min())

            if self.transform != None:
                img = self.transform(image = img)['image']

            img = torch.tensor(img, dtype = torch.float32)
            img = img.permute(2, 0, 1)
            return img
    def __len__(self):
        if self.train:
            return len(self.train_img_path)
        else:
            return len(self.test_img_path)

def dataloader(train_path, test_path, batch_size):
    Aug_train = Compose([
        albumentations.OneOf([
            albumentations.HorizontalFlip(p=1)
        ], p=0.7),
        albumentations.OneOf([
            albumentations.MotionBlur(p=1),
            albumentations.OpticalDistortion(p=1),
            albumentations.GaussNoise(p=1)
        ], p=0.7)
    ])

    train_loader = CustomDataset(True, train_path, test_path, Aug_train)
    test_loader = CustomDataset(False, train_path, test_path)

    train_ = DataLoader(train_loader, shuffle = False, batch_size = batch_size)
    test_ = DataLoader(test_loader, shuffle = False, batch_size = batch_size)
    return train_, test_