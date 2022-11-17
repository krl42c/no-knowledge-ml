from nis import cat
from random import Random
from re import I
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

import glob
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import default_collate
from PIL import Image
from PIL import ImageDraw


device = torch.device("mps")

model = torch.load("models/classifier2.pt")
model.eval()

def predict(filepath):
    img_array = Image.open(filepath).convert("RGB")
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    img = data_transforms(img_array).unsqueeze(dim=0)
    load = torch.utils.data.DataLoader(img)
    img_class = ""
    for x in load:
        x = x.to(device)
        pred = model(x)
        _, preds = torch.max(pred, 1)
        print(preds[0])
        print(f"class : {preds[0]}")
        if preds[0] == 1:
            img_class = "Dog"
        else:
            img_class = "Cat"
        print(img_class)
    img = mpimg.imread(filepath)
    imgplot = plt.imshow(img)
    plt.rc('font', size=30)          # controls default text sizes

    plt.text(0, 150, img_class)
    plt.show()

# predict("data/train/dog/dog12.jpg")

dog_samples = random.sample(range(1,49), 5)
cat_samples = random.sample(range(1,49), 5)

data_path = "data/validation/"

from os import listdir
from os.path import isfile, join

files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f.endswith(".jpg")] 
random.shuffle(files)
"""
for f in files:
    predict(data_path + f)
"""

predict("kitty.png")
