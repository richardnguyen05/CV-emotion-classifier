import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import random_split

from collections import Counter

# Note: Make sure to run preprocessing.py first before training model.

def get_class_counts(dataset):
    """
    Helper function.
    Counts the number of samples for each class within the dataset.
    Handles ImageFolder and subset objects (for validation subset)
    """
    if hasattr(dataset, "targets"):
        labels = dataset.targets
        class_names = dataset.classes
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):  # Case that object is a Subset
        labels = [dataset.dataset.targets[i] for i in dataset.indices]  # map subset indices
        class_names = dataset.dataset.classes
    else:
        raise ValueError("Dataset type not supported for get_class_counts")
    
    label_counts = Counter(labels)
    counts = [label_counts[i] for i in range(len(class_names))]
    return counts, class_names

class GrayscaleToRGBWrapper:
    """
    Helper class.
    Used to convert grayscale input into 3-channel RGB.
    This is needed for most transfer models as they are trained on RGB ImageNet
    """
    def __init__(self, dataloader): # constructor
        self.dataloader = dataloader
        self.dataset_len = len(dataloader.dataset)  # store original dataset length

    def __iter__(self):
        for images, labels in self.dataloader:
            # images: [batch_size, 1, H, W]
            images_rgb = images.repeat(1, 3, 1, 1)  # duplicate channel to 3 (RGB) [batch_size, 3, H, W]
            yield images_rgb, labels

    def __len__(self):
        return self.dataset_len

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

# splitting train dataset
train_size = int(0.8 * len(train_data)) # train will be 80% of its actual size, other 20 is for validation set
val_size = len(train_data) - train_size
train_subset, val_subset = random_split(train_data, [train_size, val_size])

# get class counts from full train data and convert to np array
counts, class_names = get_class_counts(train_data)
counts = np.array(counts, dtype=np.float32)

# WeightedRandomSampler to handle data imbalance
train_indices = train_subset.indices
targets_subset = np.array([train_data.targets[i] for i in train_subset.indices]) # targets for only the train_subset
sample_weights = 1.0 / counts[targets_subset]
sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32) # conver to tensor

# WeightedRandomSampler assigns higher sampling weights to minority classes
sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)

# creating DataLoader
raw_train_loader = DataLoader(raw_train_data, batch_size=64, shuffle=True)
train_loader = DataLoader(
    train_subset,
    batch_size= 64,
    num_workers=0,
    pin_memory=False,
    sampler=sampler,     # WeightedRandomSampler
)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False) # don't shuffle test data, for consistent evaluation

# DataLoader specific for transfer model
train_loader_rgb = GrayscaleToRGBWrapper(train_loader)
val_loader_rgb = GrayscaleToRGBWrapper(val_loader)