import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, 
                             confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support)
from train_scratch import model as model_scratch, device
from train_minixception import model as model_minix
from preprocessing import test_loader

def ModelMetrics(y_true, y_pred, model_name):
    """
    function used to calculate performance metrics

    Parameters:
        y_true : array of true labels
        y_pred : array of predicted labels
        model_name : model name being evalutaed

    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"{model_name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}") 
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    return acc, prec, rec, f1

def ModelPlots(y_true, y_pred, model_name, save_path=None):
    """
    Function to plot the confusion matrix & performance per class graph
    
    Parameters:
        y_true : array of true labels
        y_pred : array of predicted labels
        model_name : model name being evalutaed
        save_path : save path for the model plots
    
    """
    # emotion labels for FER-2013
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    display_cm.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")
    if save_path:
        plt.savefig(f"{save_path}/cm.png")
    plt.show()

    # performance per-class graph
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, prec, width, label='Precision')
    plt.bar(x, rec, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1')

    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title(f"Per-Class Performance - {model_name}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/performance_bar_graph.png")
    plt.show()

# model paths
model_minix_path = "../trained models/best_emotion_cnn_minixception.pth"
model_scratch_path = "../trained models/best_emotion_cnn_scratch.pth"

# load model weights if they exist
if os.path.exists(model_minix_path):
    state_dict = torch.load(model_minix_path, map_location=device, weights_only=True)
    model_minix.load_state_dict(state_dict)
else:
    print("MiniXception Model not found.")
if os.path.exists(model_scratch_path):
    state_dict = torch.load(model_scratch_path, map_location=device, weights_only=True)
    model_scratch.load_state_dict(state_dict)
else:
    print("Scratch Model not found.")

# set to evaluation mode
model_minix.eval()
model_scratch.eval()

# initializing prediction arrays and true labels
all_preds_minix = []
all_preds_scratch = []
all_labels = []

with torch.no_grad(): # disable gradients for test loop
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

# -- EVALUATING THE MODELS -- #
acc, prec, rec, f1 = ModelMetrics(all_labels, all_preds_minix, "MiniXception")
ModelPlots(all_labels, all_preds_minix, "MiniXception", "../plots/minixception/evaluation")

print("\n") # newline for readability

acc, prec, rec, f1 = ModelMetrics(all_labels, all_preds_scratch, "EmotionCNN (Scratch)")
ModelPlots(all_labels, all_preds_scratch, "EmotionCNN (Scratch)", "../plots/scratch/evaluation")