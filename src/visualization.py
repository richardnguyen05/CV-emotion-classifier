import os
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

from preprocessing import train_loader, raw_train_loader, train_data, test_data

def plot_samples(images, labels, class_names, n_rows=4, n_cols=4, title="Sample Images", save_path=None):
    """
    Plots a grid of images with their corresponding labels

    Parameters:
        images : torch.Tensor
            Batch of images from DataLoader, shape (batch_size, 1, H, W)
        labels : torch.Tensor
            Labels corresponding to the images
        class_names : list of str
            List of class names corresponding to label indices
        n_rows : int
            Number of rows in the subplot grid (default=4)
        n_cols : int
            Number of columns in the subplot grid (default=4)
        title : str
            Title of the plot (default="Sample Images")
    
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2)) # 4x4 subplot
    fig.suptitle(title, fontsize=14)
    for i, ax in enumerate(axes.flat): # flatten 2D array into 1D iterator, enumerate keeps track of image index
        img = images[i]
        img = img * 0.5 + 0.5 # unnormalize from [-1, 1] → [0, 1]
        img = img.squeeze() # remove channel dimension
        ax.imshow(img, cmap='gray')
        ax.set_title(class_names[labels[i]])
        ax.axis('off')
    plt.tight_layout()

    # saving the figures
    if save_path:
        # add timestamp to ensure unique file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_path.endswith(".png"): # checks if save_path already ends with png (there is already a created figure)
            full_save_path = save_path.replace(".png", f"_{timestamp}.png") # adding timestamp to ensure no overwriting figures
        else:
            full_save_path = f"{save_path}_{timestamp}.png"
        fig.savefig(full_save_path)
        print(f"Saved plot to: {full_save_path}")
    
    plt.show()
def get_class_counts(dataset):
    """
    Counts the number of samples for each class within the dataset

    Parameters:
        dataset:
            Dataset containing images and their class labels
    """
    label_counts = Counter(dataset.targets) # .targets make dataset iteration faster
    class_names = dataset.classes
    counts = [label_counts[i] for i in range(len(class_names))]
    return counts, class_names

def plot_class_distribution(dataset, title="Class Distribution"):
    """
    Plots a bar graph showing the number of samples for each class in a dataset.

    Parameters:
        dataset : torchvision.datasets.ImageFolder
            Dataset containing images and their class labels
        title : str
            Title of the plot
    """
    # get the class count and labels for each sample in the dataset
    counts, class_names = get_class_counts(dataset)

    plt.figure(figsize=(8,5))
    plt.bar(class_names, counts, color='skyblue')
    plt.xlabel("Emotion")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


# VISUALIZATIONS #

# train data before preprocessing
print("Visualizing TRAIN DATA before preprocessing...")
images, labels = next(iter(raw_train_loader))
plot_samples(images, labels, train_data.classes, title="TRAIN DATA before preprocessing", save_path="../plots/raw/raw_train_fig")

# train data after preprocessing
print("Visualizing TRAIN DATA after preprocessing...")
images, labels = next(iter(train_loader))
plot_samples(images, labels, train_data.classes, title="TRAIN DATA after preprocessing", save_path="../plots/preprocessed/preproc_train_fig")

# --- not saving class distribution plot, save_path is empty ---
# train class distribution
print("Visualizing TRAIN class distribution...")
plot_class_distribution(train_data, title="TRAIN Class Distribution")

# test class distribution
print("Visualizing TEST class distribution...")
plot_class_distribution(test_data, title="TEST Class Distribution")

print("Visualization complete.")
