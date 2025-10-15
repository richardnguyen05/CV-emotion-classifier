import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# transformations for train data
train_transform = transforms.Compose([
    transforms.Resize((48, 48)), # scales photo to 48x48
    transforms.Grayscale(num_output_channels=1), # reduces channel to 1 (grayscale, instead of the 3-channel RGB)
    transforms.RandomHorizontalFlip(), # horizontal flip (mirror)
    transforms.RandomRotation(10), # random rotation of up to +/- 10 degrees
    transforms.ColorJitter(brightness=0.3, contrast=0.3), # changes brightness and contrast of a photo sample
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.05,0.5))], p=0.2), # adds blur, wrapper transform to have a 20% probability
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