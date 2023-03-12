import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def default_loader(path):
    img = cv2.imread(path)
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return image


class pbvsDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        print ("***********************************",txt)
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
