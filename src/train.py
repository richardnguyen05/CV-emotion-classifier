import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm # progress bar for training loop

from preprocessing import train_loader, val_loader, counts


device = torch.device("cpu")  # Force to CPU usage since AMD Radeon GPU is not supported by pytorch

# --- WEIGHTED LOSS FUNCTION --- #
class_weights = 1.0 / counts # inverse freq

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) # convert to tensor
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # weight class ensures loss function prioritizes minority classes more

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

# CNN MODEL - DEEPER VERSION
class DeeperEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(DeeperEmotionCNN, self).__init__() # super calls parent class and inherits nn.Module
        
        # First Conv Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)   # 48x48 → 48x48
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 48x48 → 48x48 (same channels for greater depth at each conv block)
        self.pool1 = nn.MaxPool2d(2, 2)  # 48x48 → 24x24 (reduces spacial size by half)

        # Second Conv Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 24x24 → 24x24
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 24x24 → 24x24
        self.pool2 = nn.MaxPool2d(2, 2)  # 24x24 → 12x12

        # Third Conv Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 12x12 → 12x12
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)# 12x12 → 12x12
        self.pool3 = nn.MaxPool2d(2, 2)  # 12x12 → 6x6 

        # Fourth Conv Block
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # 6x6 → 6x6
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 6x6 → 6x6
        self.pool4 = nn.MaxPool2d(2, 2)  # 6x6 → 3x3

        # Fully Connected Layers
        # Input: 256 channels * 3x3 spatial size = 256 * 3 * 3 = 2304 features
        self.fc1 = nn.Linear(256 * 3 * 3, 1024) # reducing features to 1024
        self.fc2 = nn.Linear(1024, 512)         # additional FC layer for deeper network
        self.fc3 = nn.Linear(512, num_classes)  # produces raw logits for each of the 7 classes

        # regularlization to prevent overfitting
        self.dropout1 = nn.Dropout(0.5) # 50% dropout after first FC layer
        self.dropout2 = nn.Dropout(0.3) # 30% dropout after second FC layer

    def forward(self, x):
        # x shape: [batch_size, 1, 48, 48] - grayscale images

        # First Conv Block
        x = F.relu(self.bn1(self.conv1(x))) # [batch_size, 32, 48, 48] + batch norm
        x = F.relu(self.bn1(self.conv2(x))) # [batch_size, 32, 48, 48] + batch norm
        x = self.pool1(x)                   # [batch_size, 32, 24, 24]
        
        # Second Conv Block
        x = F.relu(self.bn2(self.conv3(x))) # [batch_size, 64, 24, 24] + batch norm
        x = F.relu(self.bn2(self.conv4(x))) # [batch_size, 64, 24, 24] + batch norm
        x = self.pool2(x)                   # [batch_size, 64, 12, 12]
        
        # Third Conv Block
        x = F.relu(self.bn3(self.conv5(x))) # [batch_size, 128, 12, 12] + batch norm
        x = F.relu(self.bn3(self.conv6(x))) # [batch_size, 128, 12, 12] + batch norm
        x = self.pool3(x)                   # [batch_size, 128, 6, 6]
        
        # Fourth Conv Block
        x = F.relu(self.bn4(self.conv7(x))) # [batch_size, 256, 6, 6] + batch norm
        x = F.relu(self.bn4(self.conv8(x))) # [batch_size, 256, 6, 6] + batch norm
        x = self.pool4(x)                   # [batch_size, 256, 3, 3]

        # flatten and fully connected
        x = torch.flatten(x, 1)                 # [batch_size, 256 * 3 * 3] = [batch_size, 2304]
        x = self.dropout1(F.relu(self.fc1(x)))  # [batch_size, 1024] + dropout
        x = self.dropout2(F.relu(self.fc2(x)))  # [batch_size, 512] + dropout
        x = self.fc3(x)                         # [batch_size, 7] - raw logits for 7 emotion classes
        
        return x

# defining the model and optimizer
model = DeeperEmotionCNN(num_classes=7).to(device) # move CNN model to device
optimizer = optim.Adam(model.parameters(), lr=0.001) # using Adam as optimizer, learning rate=0.001

# initializing variables for validation loss tracking
train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf') # start with infinitely bad loss so if-comparison in training loop works 

num_epochs = 10

# Training loop with validation
for epoch in range(num_epochs):
    model.train() # set model to training phase
    running_loss = 0.0 # reset running loss for the current epoch

    # creating progress bar
    train_pbar = tqdm(train_loader, 
                      desc=f'Epoch {epoch+1} [Train]',
                      leave=False,
                      mininterval=2.0)  # only update every 2 seconds for better performance
    
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # clear previous batch gradients
        outputs = model(images)
        loss = criterion(outputs, labels) # weighted loss value
        loss.backward() # back propogation, calculates gradient for each param and stores into .grad property
        optimizer.step() # using optimizer to adjust weights and reduce loss (opposite direction from gradient)
        running_loss += loss.item() * images.size(0)
        
        if train_pbar.n % 10 == 0:  # updates every 10 batches
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    model.eval() # set model to evaluation phase
    val_loss = 0.0
    correct = 0
    total = 0
    
    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch+1} [Val]",
        leave=False,
        mininterval=2.0  # update every 2 seconds for better performance
    )

    with torch.no_grad():  # disable gradients for validation
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # update validation progress
            if val_pbar.n % 10 == 0:
                val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
    
    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    # save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), "../trained models/best_emotion_cnn_scratch.pth")
        print(f"Best model saved with val loss: {best_val_loss:.4f}")

# save final model
torch.save(model.state_dict(), "../trained models/final_emotion_cnn_scratch.pth")
print("Training completed! Models saved.")

# Print final results
print(f"\nFinal Results:")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")




