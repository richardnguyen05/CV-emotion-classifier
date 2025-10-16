import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np

from visualization import get_class_counts

# transformations for train data
train_transform = transforms.Compose([
    transforms.Resize((48, 48)), # scales photo to 48x48
    transforms.Grayscale(num_output_channels=1), # reduces channel to 1 (grayscale, instead of the 3-channel RGB)
    transforms.RandomHorizontalFlip(), # horizontal flip (mirror)
    transforms.RandomRotation(10), # random rotation of up to +/- 10 degrees
    transforms.ColorJitter(brightness=0.3, contrast=0.3), # changes brightness and contrast of a photo sample
    transforms.ToTensor(), # converts to tensor floats
    transforms.Normalize(mean=[0.5], std=[0.5]) # normalizing tensor to [-1,1]
])
# transformations for test data (no data augmentations)
test_transform = transforms.Compose([
    transforms.Resize((48, 48)), # scales photo to 48x48
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# loading datasets
raw_train_data = datasets.ImageFolder(root='../data/raw/train', transform=test_transform)
train_data = datasets.ImageFolder(root='../data/raw/train', transform=train_transform)
test_data = datasets.ImageFolder(root='../data/raw/test', transform=test_transform)

# creating data loader
raw_train_loader = DataLoader(raw_train_data, batch_size=64, shuffle=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False) # don't shuffle test data, for consistent evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get class counts and convert to np array
counts, class_names = get_class_counts(train_data)
counts = np.array(counts, dtype=np.float32)

# inverse freq and normalization
class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum()

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) # convert to tensor

# weighted loss function & WeightedRandomSampler to handle data imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # CrossEntropyLoss ensures minority classes contribute more in the loss calculation
targets = np.array(train_data.targets)
sample_weights = 1.0 / counts[targets]
sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double) # conver to tensor

# WeightedRandomSampler assigns higher sampling weights to minority classes
sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)