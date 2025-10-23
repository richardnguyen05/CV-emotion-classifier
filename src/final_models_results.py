# learning curve, confusion matrix, and performance per class bar graph
# get performance results, and accuracy from final model and test it on test set

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score
from train_scratch import EmotionCNN, device
from train_minixception import MiniXception

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
