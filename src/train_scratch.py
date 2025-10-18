import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm # progress bar for training loop
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocessing import train_loader, val_loader, counts


device = torch.device("cpu")  # Force to CPU usage since AMD Radeon GPU is not supported by pytorch

# --- WEIGHTED LOSS FUNCTION --- #
class_weights = 1.0 / torch.sqrt(torch.tensor(counts, dtype=torch.float32)) # sqrt of inv freq weighting

class_weights_tensor = class_weights.clone().detach().to(device)
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

# defining the model and optimizer
model = EmotionCNN(num_classes=7).to(device) # move CNN model to device
optimizer = optim.Adam(model.parameters(), lr=0.001) # using Adam as optimizer, learning rate=0.001

# load previous best model and val loss if exists
best_model_path = "../trained models/best_emotion_cnn_scratch.pth"
best_val_loss_path = "../trained models/best validation loss/val_loss_scratch.txt"

# checkpoint paths
checkpoint_model_path = "../trained models/checkpoints/scratch/checkpoint_model_scratch.pth"
checkpoint_optimizer_path = "../trained models/checkpoints/scratch/checkpoint_optimizer_scratch.pth"
checkpoint_val_loss_path = "../trained models/checkpoints/scratch/checkpoint_val_loss_scratch.txt"

# load checkpoint if it exists
if os.path.exists(checkpoint_model_path) and os.path.exists(checkpoint_optimizer_path) and os.path.exists(checkpoint_val_loss_path):
    # load previous best model weights
    state_dict = torch.load(checkpoint_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    with open(checkpoint_val_loss_path, "r") as f:
        best_val_loss = float(f.read().strip())
    print(f"Loaded previous checkpoint model with val loss: {best_val_loss:.6f}")

    # load optimizer state to continue training momentum
    if os.path.exists(checkpoint_optimizer_path):
        optimizer_state = torch.load(checkpoint_optimizer_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(optimizer_state)

        print("Loaded previous optimizer state.")
    
    print("Continuing training at loaded model. To restart training, delete all contents in checkpoints scratch folder."
            "This includes:\n - checkpoint model\n - checkpoint optimizer state\n - checkpoint val loss")
else:
    # if no checkpoint exists, try to load existing best model val loss
    if os.path.exists(best_val_loss_path):
        with open(best_val_loss_path, "r") as f:
            best_val_loss = float(f.read().strip())
        print(f"No checkpoint found. Loaded existing best val loss: {best_val_loss:.6f}")
    else:
        best_val_loss = float('inf')  # starting from scratch, start with infinity loss so if-comparison in training loop works
        print("No checkpoint or previous best found. Training from scratch.")


# initializing variables for validation loss tracking
train_losses = []
val_losses = []
val_accuracies = [] # array for tracking val accuracies across all epochs

num_epochs = 15

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
    
    # save checkpoint model
    torch.save(model.state_dict(), checkpoint_model_path)
    torch.save(optimizer.state_dict(), checkpoint_optimizer_path) # enables resumed training
    with open("../trained models/checkpoints/scratch/checkpoint_val_loss_scratch.txt", "w") as f: # writing to new txt file and saving checkpoint val loss
            f.write(f"{best_val_loss:.6f}")
    # save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), best_model_path)
        with open("../trained models/best validation loss and accuracy/val_loss_scratch.txt", "w") as f: # writing to new txt file and saving best val loss
            f.write(f"{best_val_loss:.6f}")
        with open("../trained models/best validation loss and accuracy/val_accuracy_scratch.txt", "w") as f: # saving best accuracy
            best_val_accuracy = val_accuracy
            f.write(f"{best_val_accuracy:.6f}")

        print(f"Best model saved with val loss: {best_val_loss:.4f}")
        print(f"Best val loss saved in: {best_val_loss_path}")

# compute precision, recall, f1 on entire validation set
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

precision = precision_score(all_labels, all_preds, average='weighted')  # weighted accounts for class imbalance
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# print final results
print(f"\nFinal Results (from the run):")
print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")
print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")

print(f"\nFinal Results (all-time):")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1 Score: {f1:.4f}")