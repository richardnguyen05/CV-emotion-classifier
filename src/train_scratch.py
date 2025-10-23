import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm # progress bar for training loop
import matplotlib.pyplot as plt

from preprocessing import train_loader, val_loader, counts


device = torch.device("cpu")  # Force to CPU usage since AMD Radeon GPU is not supported by pytorch

# CNN MODEL
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__() # super calls parent class and inherets nn.Module
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)   # 48x48 → 48x48 (padding of 1 keeps spatial size the same)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 48x48 → 48x48
        self.pool1 = nn.MaxPool2d(2, 2)  # 48x48 → 24x24 (max pool reduces spacial size by half whilst keeping most important features)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 24x24 → 24x24
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)# 24x24 → 24x24
        self.pool2 = nn.MaxPool2d(2, 2)  # 24x24 → 12x12, max pool

        # regularization to prevent overfitting
        self.dropout_fc = nn.Dropout(0.5) # 50% of neurons are zeroed

        # Fully connected layer (linearizing 2d CNN)
        self.fc = nn.Linear(64 * 12 * 12, num_classes) # reducing features to raw logits for each of the 7 classes

    def forward(self, x):
        # x shape: [batch_size, 1, 48, 48] - grayscale images

        # First Conv Block
        x = F.relu(self.conv1(x)) # [batch_size, 32, 48, 48]
        x = F.relu(self.conv2(x)) # [batch_size, 32, 48, 48]
        x = self.pool1(x)
        # Now: [batch_size, 32, 24, 24]
        
        # Second Conv Block 
        x = F.relu(self.conv3(x)) # [batch_size, 64, 24, 24]
        x = F.relu(self.conv4(x)) # [batch_size, 64, 24, 24]
        x = self.pool2(x)
        # Now: [batch_size, 64, 12, 12]

        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        # Output: [batch_size, 7] - raw logits for 7 emotion classes
        return x

# defining the model, optimizer, and scheduler
model = EmotionCNN(num_classes=7).to(device) # move CNN model to device
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # using Adam as optimizer, weight decay reduces overfitting
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',       # minimize val loss
    factor=0.5,       # LR is multiplied by 0.5 when triggered
    patience=2,       # wait 2 epochs without improvement before reducing
    threshold=0.01,      
    threshold_mode='abs',    # absolute mode: best loss - current loss > threshold
)

# --- WEIGHTED LOSS FUNCTION --- #
class_weights = 1.0 / torch.sqrt(torch.tensor(counts, dtype=torch.float32)) # sqrt of inv freq weighting

class_weights_tensor = class_weights.clone().detach().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # weight class ensures loss function prioritizes minority classes more

# load previous best model and val loss if exists
best_model_path = "../trained models/best_emotion_cnn_scratch.pth"
best_val_loss_path = "../trained models/best validation loss/val_loss_scratch.txt"

# checkpoint paths
checkpoint_model_path = "../trained models/checkpoints/scratch/checkpoint_model_scratch.pth"
checkpoint_optimizer_path = "../trained models/checkpoints/scratch/checkpoint_optimizer_scratch.pth"
checkpoint_scheduler_path = "../trained models/checkpoints/scratch/checkpoint_scheduler_scratch.pth"
checkpoint_val_loss_path = "../trained models/checkpoints/scratch/checkpoint_val_loss_scratch.txt"

# load checkpoint if it exists
if os.path.exists(checkpoint_model_path) and os.path.exists(checkpoint_optimizer_path) and os.path.exists(checkpoint_val_loss_path) and os.path.exists(checkpoint_scheduler_path):
    # load previous checkpoint model weights
    state_dict = torch.load(checkpoint_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    scheduler_state = torch.load(checkpoint_scheduler_path, map_location=device, weights_only=True)
    scheduler.load_state_dict(scheduler_state)

    with open(checkpoint_val_loss_path, "r") as f:
        best_val_loss = float(f.read().strip())
    print(f"Loaded previous checkpoint model with val loss: {best_val_loss:.6f}")

    # load optimizer state to continue training momentum
    if os.path.exists(checkpoint_optimizer_path):
        optimizer_state = torch.load(checkpoint_optimizer_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(optimizer_state)

        print("Loaded previous optimizer state.")
    
    print("Continuing training at loaded model. To restart training, delete all contents in checkpoints SCRATCH folder."
            " This includes:\n - checkpoint model\n - checkpoint optimizer state\n - checkpoint scheduler state\n - checkpoint val loss")
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
train_accuracies = []
val_losses = []
val_accuracies = [] # array for tracking val accuracies across all epochs

num_epochs = 50

# initializing variables for early stopping
early_stopping_patience = 5
epochs_no_improve = 0
current_best_val = float('inf')

# Training loop with validation
for epoch in range(num_epochs):
    model.train() # set model to training phase
    running_loss = 0.0 # reset running loss for the current epoch
    running_corrects = 0.0 # reset running corrects for current epoch
    total_train = 0 # reset total train (total no. of samples) for current epoch

    # creating progress bar
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # clear previous batch gradients
        outputs = model(images)
        loss = criterion(outputs, labels) # weighted loss value
        loss.backward() # back propogation, calculates gradient for each param and stores into .grad property
        optimizer.step() # using optimizer to adjust weights and reduce loss (opposite direction from gradient)
        running_loss += loss.item() * images.size(0)
        
        # calculating train accuracy for this batch
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total_train += labels.size(0)

        # update progress bar
        train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / total_train * 100
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    model.eval() # set model to evaluation phase
    val_loss = 0.0
    correct = 0
    total = 0
    
    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch+1} [Val]", leave=False)

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
            
            # update progress bar
            val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
    
    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total * 100
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_accuracy)
    scheduler.step(val_epoch_loss) # step scheduler based on validation loss
    print("Current LR:", scheduler.get_last_lr()) # print current lr
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    # save checkpoint model
    torch.save(model.state_dict(), checkpoint_model_path)
    torch.save(optimizer.state_dict(), checkpoint_optimizer_path)
    torch.save(scheduler.state_dict(), checkpoint_scheduler_path)
    with open("../trained models/checkpoints/scratch/checkpoint_val_loss_scratch.txt", "w") as f: # writing to new txt file and saving checkpoint val loss
            f.write(f"{val_epoch_loss:.6f}")
    
    # check if current epoch greater than best current epoch
    if val_epoch_loss < current_best_val:
        current_best_val = val_epoch_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    # save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), best_model_path)
        with open("../trained models/best validation loss/val_loss_scratch.txt", "w") as f: # writing to new txt file and saving best val loss
            f.write(f"{best_val_loss:.6f}")
    
        print(f"Best model saved with val loss: {best_val_loss:.4f}")
        print(f"Best val loss saved in: {best_val_loss_path}")

    # Early stopping check
    if epochs_no_improve >= early_stopping_patience:
        print(f"Early stopping triggered. No improvement in val loss for {early_stopping_patience} epochs.")
        break

epochs = range(1, len(train_losses) + 1) # list of epochs
# plot final results

# ----- Loss Plot -----
plt.figure(figsize=(8,5))
plt.plot(epochs, train_losses, label='Train Loss', color='blue', linestyle='-')
plt.plot(epochs, val_losses, label='Val Loss', color='red', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("../plots/scratch/loss.png")
plt.show()

# ----- Accuracy Plot -----
plt.figure(figsize=(8,5))
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue', linestyle='-')
plt.plot(epochs, val_accuracies, label='Val Accuracy', color='red', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig("../plots/scratch/acc.png")
plt.grid(True)
plt.show()