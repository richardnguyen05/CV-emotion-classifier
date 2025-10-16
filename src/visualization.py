import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

from preprocessing import train_loader, raw_train_loader, train_data, test_data, val_subset

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
    for i, ax in enumerate(axes.flat):
        img = images[i]
        img = img * 0.5 + 0.5 # unnormalize from [-1, 1] → [0, 1]
        img = img.squeeze() # remove channel dimension
        ax.imshow(img, cmap='gray')
        ax.set_title(class_names[labels[i]])
        ax.axis('off')
    plt.tight_layout()

    if save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_path.endswith(".png"):
            full_save_path = save_path.replace(".png", f"_{timestamp}.png")
        else:
            full_save_path = f"{save_path}_{timestamp}.png"
        fig.savefig(full_save_path)
        print(f"Saved plot to: {full_save_path}")
    
    plt.show()

def get_class_counts(dataset):
    """
    Counts the number of samples for each class within the dataset
    Handles ImageFolder and Subset objects
    """
    if hasattr(dataset, "targets"):
        labels = dataset.targets
        class_names = dataset.classes
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):  # HERE: Subset
        labels = [dataset.dataset.targets[i] for i in dataset.indices]  # map subset indices
        class_names = dataset.dataset.classes
    else:
        raise ValueError("Dataset type not supported for get_class_counts")
    
    label_counts = Counter(labels)
    counts = [label_counts[i] for i in range(len(class_names))]
    return counts, class_names

def plot_class_distribution(dataset, title="Class Distribution", save_path=None):
    """
    Plots a bar graph showing the number of samples for each class in a dataset.
    """
    counts, class_names = get_class_counts(dataset)
    plt.figure(figsize=(8,5))
    plt.bar(class_names, counts, color='skyblue')
    plt.xlabel("Emotion")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved fig to: {save_path}")
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

# train class distribution
print("Visualizing TRAIN class distribution...")
plot_class_distribution(train_data, title="TRAIN Class Distribution", save_path="../plots/class distribution/train_class_dis_fig")

# val class distribution
print("Visualizing VAL class distribution...")
plot_class_distribution(val_subset, title="VAL Class Distribution", save_path="../plots/class distribution/val_class_dis_fig")

# test class distribution
print("Visualizing TEST class distribution...")
plot_class_distribution(test_data, title="TEST Class Distribution", save_path="../plots/class distribution/test_class_dis_fig")

print("Visualization complete.")
