import matplotlib as plt
def matplotlib_imshow(img_batch):
    # Extract the first image from the batch
    img = img_batch[0]
    # If the image has a single channel, squeeze it to 2D
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)

    plt.imshow(img, cmap='gray')
    plt.show()
