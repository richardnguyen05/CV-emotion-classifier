import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from preprocessing import train_loader, val_loader, counts
from train_scratch import criterion # use same loss function as scratch model

device = torch.device("cpu")  # Force to CPU usage since AMD Radeon GPU is not supported by pytorch

class EmotionResNet18(nn.Module):
    def __init__(self, num_classes=7, freeze_backbone=True):
        super(EmotionResNet18, self).__init__()

        # load pretrained ResNet18 model
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # freeze feature extractor, weights won't change since model already learned general features
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # conv layer 1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # modified input to accept grayscale instead of RGB
        # output feature map = [24, 24]

        # replace the classifier (fc layer)
        in_features = self.backbone.fc.in_features # for ResNet18, feature map is compressed to single vector of 512
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256), # 512 → 256, reduce output to 256 channels
            nn.ReLU(), # adds non-linearity
            nn.Dropout(0.3), # randomly drops 30% of neurons during training
            nn.Linear(256, num_classes) # 256 → 7, logits for the emotion classes
        )

    def forward(self, x):
        # forward pass through modified ResNet
        return self.backbone(x)

# initializing model and optimizer
model = EmotionResNet18(num_classes=7, freeze_backbone=True).to(device)
optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=1e-4) # backbone frozen, only classifier is trained

# unfreeze last residual layer after classifier converges (ResNet18 has 4 residual layers, each layer having 4 conv layers and 2 basic blocks)
for name, param in model.backbone.named_parameters():
    if "layer4" in name:
        param.requires_grad = True # unfreezing all of backbone risks overfitting for small-medium sized datasets

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5) # lower lr prevents destroying weights during fine-tuning

# load previous best model and val loss if exists
best_model_path = "../trained models/best_emotion_cnn_transfer.pth"
best_val_loss_path = "../trained models/best validation loss/val_loss_transfer.txt"

# checkpoint paths
checkpoint_model_path = "../trained models/checkpoints/transfer/checkpoint_model_transfer.pth"
checkpoint_optimizer_path = "../trained models/checkpoints/transfer/checkpoint_optimizer_transfer.pth"
checkpoint_val_loss_path = "../trained models/checkpoints/transfer/checkpoint_val_loss_transfer.txt"

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
    
    print("Continuing training at loaded model. To restart training, delete all contents in checkpoints TRANSFER folder."
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

# training loop with validation
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

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

        # update progress bar
        train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss) # add the epoch loss to the train array

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

    print(f"Epoch [{epoch+1/{num_epochs}}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}")

    # save checkpoint model
    torch.save(model.state_dict(), checkpoint_model_path)
    torch.save(optimizer.state_dict(), checkpoint_optimizer_path) # enables resumed training
    with open("../trained models/checkpoints/transfer/checkpoint_val_loss_transfer.txt", "w") as f: # writing to new txt file and saving checkpoint val loss
            f.write(f"{val_epoch_loss:.6f}")
    # save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), best_model_path)
        with open("../trained models/best validation loss/val_loss_transfer.txt", "w") as f: # writing to new txt file and saving best val loss
            f.write(f"{best_val_loss:.6f}")
    
        print(f"Best model saved with val loss: {best_val_loss:.4f}")
        print(f"Best val loss saved in: {best_val_loss_path}")

# print final results
print(f"\nFinal Results of the Run:")
print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")
print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
print(f"All Validation Accuracies: {val_accuracies}")

print(f"\nLosses for all epochs:")
print(f"Train: {train_losses}")
print(f"Validation: {val_losses}")