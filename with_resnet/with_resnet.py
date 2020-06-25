from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import os
from matplotlib import pyplot as plt
import numpy as np
from torch import nn
from torch.nn import functional as F
from random import randint
from pytorch_lightning.callbacks import ModelCheckpoint


# 20 class
all_data = ['churros', 'hamburger', 'omelette', 'cup_cakes', 'greek_salad',
            'lasagna', 'seaweed_salad', 'edamame', 'sushi', 'ramen', 'caesar_salad',
            'chocolate_mousse', 'hot_dog', 'ice_cream', 'beet_salad', 'donuts',
            'bibimbap', 'pizza', 'grilled_cheese_sandwich', 'grilled_salmon']


class CustomDataSet(Dataset):
    def __init__(self, dir, validation=False):
        self.dir = dir
        self.validation = validation
        self.images = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.augmentation = [transforms.CenterCrop(250), transforms.Pad(5), transforms.RandomHorizontalFlip(p=1),
                             transforms.RandomRotation((-50, 50)), transforms.RandomVerticalFlip(p=1),
                             transforms.RandomCrop(250),
                             transforms.Compose([
                                 transforms.RandomVerticalFlip(p=1),
                                 transforms.RandomHorizontalFlip(p=1)
                             ]),
                             transforms.Compose([
                                 transforms.RandomVerticalFlip(p=1),
                                 transforms.RandomHorizontalFlip(p=1),
                                 transforms.Pad(5)
                             ]),
                             transforms.Compose([
                                 transforms.RandomVerticalFlip(p=1),
                                 transforms.Pad(5)
                             ]),
                             transforms.Compose([
                                 transforms.RandomHorizontalFlip(p=1),
                                 transforms.Pad(5)
                             ]),
                             transforms.Compose([
                                 transforms.RandomHorizontalFlip(p=1),
                                 transforms.RandomRotation((-50, 50)),
                                 transforms.CenterCrop(250)
                             ]),
                             transforms.Compose([
                                 transforms.CenterCrop(250),
                                 transforms.Pad(5)
                             ]),
                             transforms.Compose([
                                 transforms.RandomRotation((-50, 50)),
                                 transforms.RandomCrop(250),
                                 transforms.Pad(5)
                             ]),
                             transforms.Compose([
                                 transforms.RandomHorizontalFlip(p=1),
                                 transforms.RandomRotation((-50, 50)),
                                 transforms.RandomCrop(250)
                             ]),
                             transforms.Compose([
                                 transforms.RandomRotation((-50, 50)),
                                 transforms.RandomCrop(250)
                             ]),
                             transforms.Compose([
                                 transforms.RandomRotation((-50, 50)),
                                 transforms.CenterCrop(250)
                             ]),
                             transforms.Compose([
                                 transforms.RandomVerticalFlip(p=1),
                                 transforms.RandomHorizontalFlip(p=1),
                                 transforms.RandomCrop(250)
                             ])
                            ]

        for dirs in os.listdir(self.dir):
            photo_path_in_dir = dirs
            for images in os.listdir(os.path.join(self.dir, photo_path_in_dir)):
                self.images += [os.path.join(photo_path_in_dir, images)]
                self.labels += [np.array(all_data.index(photo_path_in_dir))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.dir, self.images[i])).convert('RGB')
        if randint(0, 1) == 1 and (not self.validation):
            image = image.resize((300, 300))
            image = transforms.RandomChoice(self.augmentation)(image)
        image = image.resize((224, 224))

        return self.transform(image), torch.from_numpy(self.labels[i])


class Food_Net(LightningModule):
    def __init__(self):
        super().__init__()
        # 1000 output
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        for name, param in self.model.named_parameters():
            if("bn" not in name):
                param.requires_grad = False

        # unfreeze last layer
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        # unfreeze pre-last layer
        for param in self.model.layer3.parameters():
            param.requires_grad = True

        # changing classifier
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 20),
            nn.LogSoftmax(dim=1)
        )

        self.valid = 0
        self.set_size = 0

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        ds_train = CustomDataSet("./dataset/train")
        return DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self):
        ds_val = CustomDataSet("./dataset/valid", validation=True)
        self.set_size = len(ds_val)
        print("Valid set size:", self.set_size)
        return DataLoader(ds_val, batch_size=32, num_workers=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        print("\n\nEPOCH train_loss:", avg_loss, "\n")
        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        for i in range(len(y_hat)):
            idx = torch.argmax(y_hat[i])
            if idx == batch[1][i]:
                self.valid += 1
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        print("\n\nValidation accuracy:", (self.valid/self.set_size))
        self.valid = 0

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        print("EPOCH val_loss:", avg_loss, "\n")
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        filepath='./model_resnet_semi_unfreeze/',
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    trainer = Trainer(gpus=1, log_save_interval=1,
                      checkpoint_callback=checkpoint_callback)
    net = Food_Net()
    trainer.fit(net)
