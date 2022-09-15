from nis import cat
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import default_collate

import shutil

cat_sample = cv2.imread("kagglecatsanddogs_5340/PetImages/Cat/0.jpg")
cat_sample = cv2.resize(cat_sample, (100, 100), interpolation=cv2.INTER_AREA)
plt.clf()
#imgplot = plt.imshow(cat_sample)

# Load data

cat_photos = []
dog_photos = []

device = torch.device("mps")


def load_files(cat_photos, dog_photos, path):
    transform = transforms.ToTensor()
    print("Loading cat files")

    path_cat = path + "/cat*"
    path_dog = path + "/dog+"

    for file in glob.iglob(path_cat):
        img = cv2.imread(file)
        if img is not None:
            img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
            tensor = transform(img)
            cat_photos.append(tensor)

    for file in glob.iglob(path_dog):
        print("working?")
        img = cv2.imread(file)
        if img is not None:
            img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
            dog_photos.append(img)

# Prepare data


def copy_files():
    i = 0
    class_name = "cat"
    folder = "data/train/"

    for file in glob.iglob("kagglecatsanddogs_5340/PetImages/Cat/*.jpg"):
        if i == 50:
            break
        dest = folder + class_name + str(i)
        print("Copying " + dest)
        shutil.copy(file, dest)
        i = i + 1

    i = 0
    class_name = "dog"
    for file in glob.iglob("kagglecatsanddogs_5340/PetImages/Dog/*.jpg"):
        if i == 50:
            break
        dest = folder + class_name + str(i)
        print("Copying " + dest)
        shutil.copy(file, dest)
        i = i + 1



transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder("data/train/", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=lambda x: tuple(
    x_.to(device) for x_ in default_collate(x)), shuffle=True)
print("OKOKOK")
labels = {
    0: "cat",
    1: "dog"
}

images, labels = next(iter(dataloader))

dataiter = iter(dataloader)
images, labels = dataiter.next()

model_resnet = models.densenet121(pretrained=True)
model_resnet.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 512),
                                              torch.nn.ReLU(),
                                              torch.nn.Dropout(0.2),
                                              torch.nn.Linear(512, 256),
                                              torch.nn.ReLU(),
                                              torch.nn.Dropout(0.1),
                                              torch.nn.Linear(256, 2),
                                              torch.nn.LogSoftmax(dim=1))

model_resnet.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_resnet.parameters(), lr=0.001)
for epoch in range(2):
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs.to(device)
        labels.to(device)

        outputs = model_resnet(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        print(f"Current loss {loss.item()}")

torch.save(model_resnet, "models/classifier.pt")
