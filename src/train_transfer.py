import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from preprocessing import train_loader, val_loader, counts

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

# --- WEIGHTED LOSS FUNCTION --- #
class_weights = 1.0 / torch.sqrt(torch.tensor(counts, dtype=torch.float32)) # sqrt of inv freq weighting

class_weights_tensor = class_weights.clone().detach().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # weight class ensures loss function prioritizes minority classes more
