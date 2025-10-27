# learning curve, confusion matrix, and performance per class bar graph
# get performance results, and accuracy from final model and test it on test set

import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from train_scratch import EmotionCNN, device
from train_minixception import MiniXception
from preprocessing import test_loader

# recreating models
model_minix = MiniXception(num_classes=7).to(device)
model_scratch = EmotionCNN(num_classes=7).to(device)

# model paths
model_minix_path = "../trained models/best_emotion_cnn_minixception.pth"
model_scratch_path = "../trained models/best_emotion_cnn_scratch.pth"

# load model weights
state_dict = torch.load(model_minix_path, map_location=device, weights_only=True)
model_minix.load_state_dict(state_dict)
state_dict = torch.load(model_scratch_path, map_location=device, weights_only=True)
model_scratch.load_state_dict(state_dict)

model_minix.eval()
model_scratch.eval()

# initializing prediction arrays and true labels
all_preds_minix = []
all_preds_scratch = []
all_labels = []

with torch.no_grad: # disable gradients
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs_minix = model_minix(images)
        outputs_scratch = model_scratch(images)

        # get predicted classes
        preds_minix = torch.argmax(outputs_minix, dim=1)
        preds_scratch = torch.argmax(outputs_scratch, dim=1)

        # store predictions and true labels
        all_preds_minix.extend(preds_minix.cpu().numpy())
        all_preds_scratch.extend(preds_scratch.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# convert to numpy arrays
all_preds_minix = np.array(all_preds_minix)
all_preds_scratch = np.array(all_preds_scratch)
all_labels = np.array(all_labels)

# compute metrics
