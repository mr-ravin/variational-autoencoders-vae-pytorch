import glob
import cv2
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, mode="train"):
        self.net_data = []
        if mode=="train":
            self.path = "dataset/train/"
        elif mode=="valid":
            self.path = "dataset/valid/"
        elif mode=="test":
            self.path = "dataset/test/"
        self.net_data = glob.glob(self.path+"*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.net_data)

    def __getitem__(self, idx):
        img_path = self.net_data[idx]
        image = cv2.imread(img_path)
        image = image/255.0
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, img_path