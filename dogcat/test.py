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

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder("data/train/", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=lambda x: tuple(
    x_.to(device) for x_ in default_collate(x)), shuffle=True)


model = torch.load("models/classifier.pt")
model.eval()

pred = []
with torch.no_grad():
    for images, labels in dataloader:
        output = model(images)
        _, prediction = torch.max(output, 1)
        pred.append(prediction)

        print(f"Actual\n{labels}")
        print(f"Predicted\n{prediction}")


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
            print(f"predicted ----> Dog")
            img_class = "DOG predicted"
        else:
            print(f"predicted ----> Cat")
            img_class = "CAT predicted"
    img = mpimg.imread(filepath)
    imgplot = plt.imshow(img)
    plt.rc('font', size=30)          # controls default text sizes

    plt.text(200, 150, img_class)
    plt.show()

# predict("data/train/dog/dog12.jpg")

dog_samples = random.sample(range(1,49), 5)
cat_samples = random.sample(range(1,49), 5)

dog_folder = "data/train/dog/"
cat_folder = "data/train/cat/"

for x in dog_samples:
    file = dog_folder + "dog" + str(x) + ".jpg"
    predict(file)

for x in cat_samples:
    file = cat_folder + "cat" + str(x) + ".jpg"
    predict(file)
