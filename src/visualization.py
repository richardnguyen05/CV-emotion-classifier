import matplotlib.pyplot as plt
from collections import Counter

from preprocessing import train_loader, raw_train_loader, train_data, test_data

def plot_samples(images, labels, class_names, n_rows=4, n_cols=4, title="Sample Images"):
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
    plt.show()

def plot_class_distribution(dataset, title="Class Distribution"):
    """
    Plots a bar graph showing the number of samples for each class in a dataset.

    Parameters:
        dataset : torchvision.datasets.ImageFolder
            Dataset containing images and their class labels
        title : str
            Title of the plot
    """
    # Count the number of samples for each class
    label_counts = Counter(dataset.targets) # .targets make dataset iteration faster
    class_names = dataset.classes
    counts = [label_counts[i] for i in range(len(class_names))]

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
plot_samples(images, labels, train_data.classes, title="TRAIN DATA before preprocessing")

# train data after preprocessing
print("Visualizing TRAIN DATA after preprocessing...")
images, labels = next(iter(train_loader))
plot_samples(images, labels, train_data.classes, title="TRAIN DATA after preprocessing")

# train class distribution
print("Visualizing TRAIN class distribution...")
plot_class_distribution(train_data, title="TRAIN Class Distribution")

# test class distribution
print("Visualizing TEST class distribution...")
plot_class_distribution(test_data, title="TEST Class Distribution")
