import matplotlib.pyplot as plt
from collections import Counter

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
        img = images[i].squeeze(0) # remove channel dimension for grayscale
        img = img * 0.5 + 0.5 # unnormalize from [-1, 1] → [0, 1]
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
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    
    # Get class names
    class_names = dataset.classes
    
    # Prepare data for plotting
    counts = [label_counts[i] for i in range(len(class_names))]
    
    # Plot bar graph
    plt.figure(figsize=(8,5))
    plt.bar(class_names, counts, color='skyblue')
    plt.xlabel("Emotion")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    from preprocessing import raw_train_data, train_data, train_loader, test_data

    print("Visualizing TRAIN DATA before preprocessing...")
    # raw samples 
    raw_images, raw_labels = zip(*[raw_train_data[i] for i in range(16)])  # get first 16 samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Raw Training Samples (Before Preprocessing)", fontsize=14)
    for i, ax in enumerate(axes.flat):
        img, label = raw_images[i], raw_labels[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(raw_train_data.classes[label])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    print("Visualizing TRAIN DATA after preprocessing...")
    # preprocessed (augmented) samples
    images, labels = next(iter(train_loader))
    plot_samples(images, labels, train_data.classes, title="Training Samples (After Preprocessing)")

    print("Visualizing TRAIN and TEST class distributions...")
    plot_class_distribution(raw_train_data, title="Train Class Distribution")
    plot_class_distribution(test_data, title="Test Class Distribution")

    print("Visualization complete.")