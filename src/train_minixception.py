import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from preprocessing import train_loader, val_loader, counts

device = torch.device("cpu")  # Force to CPU usage 

# Depthwise Separable Convolution 
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                                   groups=in_ch, bias=False) # depthwise convolution applied to each input channel independently
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False) # pointwise conv combines outputs linearly
        self.bn = nn.BatchNorm2d(out_ch) # batch normalization
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

# MiniXception Architecture 
class MiniXception(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # entry block
        # conv block 1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8), # batch normalization
            nn.ReLU(inplace=True) # ReLU activation
        ) # conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # middle block (residual depthwise-separable blocks)
        self.block1 = self._residual_block(8, 16) # each residual block has 2 SeparableConv blocks
        self.block2 = self._residual_block(16, 32)
        self.block3 = self._residual_block(32, 64)
        self.block4 = self._residual_block(64, 128)

        # exit block
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3) # drop 30% of neurons
        self.fc = nn.Linear(128, num_classes) # fc layer to produce raw logits for emotion classes

    def _residual_block(self, in_ch, out_ch):
        """

        Defines the depthwise-separable residual block in the MiniXception model.
        Main applies two depthwise separable convolutions and reduces spatial size by half.
        Residual is the original input convolved to match main's spatial size.

        """
        residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False) # orignal input with 1x1 convolution and stride 2
        main = nn.Sequential(
            SeparableConv2d(in_ch, out_ch),
            SeparableConv2d(out_ch, out_ch),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # reduces spacial size
        )
        return nn.Sequential(nn.ModuleDict({'main': main, 'residual': residual}))

    def forward(self, x):
        # entry
        x = self.conv1(x)
        x = self.conv2(x)

        # residual connections manually applied
        for block in [self.block1, self.block2, self.block3, self.block4]:
            main = block[0]['main'](x)
            residual = block[0]['residual'](x)
            x = main + residual  # residual connection added to main

        # exit
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# initializing model and optimizer
model = MiniXception(num_classes=7).to(device) # depthwise separable with residual connections
# optimizer only for classifier initially
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# lr scheduler (reduce lr gradually in a cosine curve)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=25, eta_min=1e-5
) # eta_min is the final lr at end of curve

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
checkpoint_scheduler_path = "../trained models/checkpoints/minixception/checkpoint_scheduler_minixception.pth"
checkpoint_val_loss_path = "../trained models/checkpoints/minixception/checkpoint_val_loss_minixception.txt"

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
    
    print("Continuing training at loaded model. To restart training, delete all contents in checkpoints MINIXCEPTION folder."
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
    scheduler.step()
    print("Current LR:", scheduler.get_last_lr())

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # save checkpoint model
    torch.save(model.state_dict(), checkpoint_model_path)
    torch.save(optimizer.state_dict(), checkpoint_optimizer_path)
    torch.save(scheduler.state_dict(), checkpoint_scheduler_path)
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