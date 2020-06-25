from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import os
from matplotlib import pyplot as plt
import numpy as np
from torch import nn
from pytorch_lightning import LightningModule


# 20 class
all_data = ['churros', 'hamburger', 'omelette', 'cup_cakes', 'greek_salad',
            'lasagna', 'seaweed_salad', 'edamame', 'sushi', 'ramen', 'caesar_salad',
            'chocolate_mousse', 'hot_dog', 'ice_cream', 'beet_salad', 'donuts',
            'bibimbap', 'pizza', 'grilled_cheese_sandwich', 'grilled_salmon']

# 2 class
all_answ = [0, 0, 1, 0, 1,
            0, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            0, 1, 0, 0, 1]


class CustomDataSet(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.images = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for dirs in os.listdir(self.dir):
            photo_path_in_dir = dirs
            for images in os.listdir(os.path.join(self.dir, photo_path_in_dir)):
                self.images += [os.path.join(photo_path_in_dir, images)]
                self.labels += [np.array(all_data.index(photo_path_in_dir))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.dir, self.images[i])).convert('RGB')
        image = image.resize((224, 224))

        return self.transform(image), torch.from_numpy(self.labels[i])


if __name__ == "__main__":
    model = torch.load("./with_resnet.model")
    ds_val = CustomDataSet("/home/pan_orka/Code/Python/machine_learning/solvro_test/dataset/valid")
    model.eval()
    with torch.no_grad():
        ctr = 0
        for x, y in ds_val:
            if torch.argmax(model(x.unsqueeze(0))) == y:
                ctr += 1
        print("Accuracy is (20 class):", ctr/len(ds_val))
        ctr = 0
        for x, y in ds_val:
            if all_answ[torch.argmax(model(x.unsqueeze(0))).item()] == all_answ[y.item()]:
                ctr += 1
        print("Accuracy is (2 class):", ctr/len(ds_val))
