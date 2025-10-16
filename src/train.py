import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt

from preprocessing import train_loader
from visualization import counts


device = torch.device("cpu")  # Force to CPU usage since AMD Radeon GPU is not supported 

# --- GETTING WEIGHTED LOSS FUNCTION --- #
# inverse freq and normalization
class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum()

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) # convert to tensor
# CrossEntropyLoss ensures minority classes contribute more in the loss calculation
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# CNN MODEL
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__() # super calls parent class and inherets nn.Module
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)   # 48x48 → 48x48 (padding of 1 keeps spatial size the same)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 48x48 → 48x48
        self.pool1 = nn.MaxPool2d(2, 2)  # 48x48 → 24x24 (pool1 reduces spacial size by half)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 24x24 → 24x24
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)# 24x24 → 24x24
        self.pool2 = nn.MaxPool2d(2, 2)  # 24x24 → 12x12

        # Fully connected layers (linearizing 2d CNN)
        self.fc1 = nn.Linear(128 * 12 * 12, 512) # reducing features to 512
        self.fc2 = nn.Linear(512, num_classes) # produces raw logits for each of the 7 classes

        # Dropout for regularization to prevent overfitting
        self.dropout = nn.Dropout(0.5) # 50% of neurons are zeroed

    def forward(self, x):
        # x shape: [batch_size, 1, 48, 48] - grayscale images

        # First Conv Block
        x = F.relu(self.conv1(x)) # [batch_size, 32, 48, 48]
        x = F.relu(self.conv2(x)) # [batch_size, 64, 48, 48]
        x = self.pool1(x)
        # Now: [batch_size, 64, 24, 24]
        
        # Second Conv Block 
        x = F.relu(self.conv3(x)) # [batch_size, 128, 24, 24]
        x = F.relu(self.conv4(x)) # [batch_size, 128, 24, 24]
        x = self.pool2(x)
        # Now: [batch_size, 128, 12, 12]

        x = torch.flatten(x, 1) # [batch_size, 128 * 12 * 12]
        x = self.dropout(F.relu(self.fc1(x))) # [batch_size, 512]
        x = self.fc2(x)
        # Output: [batch_size, 7] - raw logits for 7 emotion classes
        return x

# defining the model and optimizer
model = EmotionCNN(num_classes=7).to(device) # move CNN model to device
optimizer = optim.Adam(model.parameters(), lr=0.001) # using Adam as optimizer

num_epochs = 10
# training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0 # reset running loss for the current epoch
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # clear previous batch gradients
        outputs = model(images)
        loss = criterion(outputs, labels)  # weighted loss
        loss.backward()
        optimizer.step() # using optimizer to adjust weights

        running_loss += loss.item() * images.size(0) # track accumulating loss

    epoch_loss = running_loss / len(train_loader.dataset) # normalize loss by total samples, and store that loss according to its epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


# ADD CODE FOR SAVIVG MODEL HERE (SAVE MODEL BASED ON VALIDATION LOSS)

