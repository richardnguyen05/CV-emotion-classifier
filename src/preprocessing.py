import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

# transformations for train data
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(), # converts RGB to tensor floats
    transforms.Normalize(mean=[0.5], std=[0.5]) # normalizing tensor to [-1,1]
])
# transformations for test data
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# loading datasets
train_data = datasets.ImageFolder(root='../data/raw/train', transform=train_transform)
test_data = datasets.ImageFolder(root='../data/raw/test', transform=test_transform)

# creating data loader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False) # don't shuffle test data, for consistent evaluation

# visualizing train data
images, labels = next(iter(train_loader))
img = images[0].squeeze() * 0.5 + 0.5   # unnormalize [-1,1] → [0,1]
plt.imshow(img, cmap='gray')
plt.title(f"Label: {train_data.classes[labels[0]]}")
plt.show()