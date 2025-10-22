import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from preprocessing import train_loader, val_loader, counts

device = torch.device("cpu")  # Force to CPU usage since AMD Radeon GPU is not supported by pytorch

class MiniXception(nn.Module):
    def __init__(self, num_classes=7):
        super(MiniXception, self).__init__()

        def conv_block(in_ch, out_ch, pool=True): # defining a convolution block. we have 2 conv layers per block, set pool to true to enable max pool
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1), # kernel size of 3, padding of 1 keeps spatial size same
                nn.BatchNorm2d(out_ch), # batch normalization
                nn.ReLU(inplace=True), # ReLU Activation (repeat the three steps again to form the conv block)
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3) # drop 30% of neurons during training
            ]
            if pool:
                layers.append(nn.MaxPool2d(2)) # reduce spacial size by half after each block
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1, 8),   # input is grayscale 1 channel
            conv_block(8, 16),
            conv_block(16, 32),
            conv_block(32, 64) # with each conv block, feature map is doubled
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1) # global pooling
        self.fc = nn.Linear(64, num_classes) # fc layer produces raw logits (7) for emotion classes

    def forward(self, x):
        x = self.features(x) # pass input through all conv blocks
        x = self.global_pool(x) # apply global pool, reducing feature map to 1x1 shape
        x = x.view(x.size(0), -1) # flatten, so we can pass it to fc layer
        x = self.fc(x)
        return x

# initializing model and optimizer
model = MiniXception(num_classes=7).to(device)
# optimizer only for classifier initially
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# lr scheduler (reduce lr if it plateaus). same scheduler as scratch custom CNN
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',       # minimize val loss
    factor=0.5,       # LR is multiplied by 0.5 when triggered
    patience=2,       # wait 2 epochs without improvement before reducing
    threshold=0.01,
    threshold_mode='abs',
)

# --- WEIGHTED LOSS FUNCTION --- #
class_weights = 1.0 / torch.sqrt(torch.tensor(counts, dtype=torch.float32)) # sqrt of inv freq weighting

class_weights_tensor = class_weights.clone().detach().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # weight class ensures loss function prioritizes minority classes more

# load previous best model and val loss if exists
best_model_path = "../trained models/best_emotion_cnn_minixception.pth"
best_val_loss_path = "../trained models/best validation loss/val_loss_minixception.txt"

# checkpoint paths
checkpoint_model_path = "../trained models/checkpoints/minixception/checkpoint_model_minixception.pth"
checkpoint_optimizer_path = "../trained models/checkpoints/minixception/checkpoint_optimizer_minixception.pth"
checkpoint_val_loss_path = "../trained models/checkpoints/minixception/checkpoint_val_loss_minixception.txt"

# load checkpoint if it exists
if os.path.exists(checkpoint_model_path) and os.path.exists(checkpoint_optimizer_path) and os.path.exists(checkpoint_val_loss_path):
    # load previous checkpoint model weights
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
    
    print("Continuing training at loaded model. To restart training, delete all contents in checkpoints MINIXCEPTION folder."
            " This includes:\n - checkpoint model\n - checkpoint optimizer state\n - checkpoint val loss")
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

num_epochs = 25
epochs = range(1, num_epochs + 1) # list of epochs

# training loop with validation
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train = 0

    # progress bar
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device) # moving images and labels to device
        optimizer.zero_grad() # clearing previous gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # takes loss value and calculates gradient, storing into .grad property
        optimizer.step() # use optimizer to reduce loss (move opposite to gradient)
        running_loss += loss.item() * images.size(0)

        # calculating train accuracy for this batch
        _, preds = torch.max(outputs, 1)
        running_corrects += (preds == labels).sum().item()
        total_train += labels.size(0)

        # update progress bar
        train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / total_train * 100
    train_losses.append(epoch_loss) # add the epoch loss to the train array
    train_accuracies.append(epoch_acc)

    model.eval() # evaluation phase
    val_loss = 0.0
    correct = 0
    total = 0

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0) # no of samples in batch
            correct += (predicted == labels).sum().item() # if predicted = labels, sum all correct predictions

            val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total * 100
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_accuracy)
    scheduler.step(val_epoch_loss)
    print("Current LR:", scheduler.get_last_lr())

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # save checkpoint model
    torch.save(model.state_dict(), checkpoint_model_path)
    torch.save(optimizer.state_dict(), checkpoint_optimizer_path) # enables resumed training
    with open("../trained models/checkpoints/minixception/checkpoint_val_loss_minixception.txt", "w") as f: # writing to new txt file and saving checkpoint val loss
            f.write(f"{val_epoch_loss:.6f}")
    # save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), best_model_path)
        with open("../trained models/best validation loss/val_loss_minixception.txt", "w") as f: # writing to new txt file and saving best val loss
            f.write(f"{best_val_loss:.6f}")
    
        print(f"Best model saved with val loss: {best_val_loss:.4f}")
        print(f"Best val loss saved in: {best_val_loss_path}")

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
plt.savefig("../plots/minixception/loss.png") # save fig to plots folder
plt.show()

# ----- Accuracy Plot -----
plt.figure(figsize=(8,5))
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue', linestyle='-')
plt.plot(epochs, val_accuracies, label='Val Accuracy', color='red', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig("../plots/minixception/acc.png") # save fig to plots folder
plt.grid(True)
plt.show()