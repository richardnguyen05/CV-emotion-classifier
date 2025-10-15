import matplotlib.pyplot as plt
from preprocessing import train_loader, train_data 


# visualizing train data
images, labels = next(iter(train_loader))



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
