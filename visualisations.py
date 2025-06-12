import matplotlib.pyplot as plt


def visualise_10_images(ds):
    """Show ten images in the dataset. Need to build the dataset first

    Args:
        ds (tf dataset)
    """
    image_batch, label_batch = next(iter(ds))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy())
        label = label_batch[i].numpy()
        plt.title(
            "lat: " + str(round(label[0], 2)) + ", long: " + str(round(label[1], 2))
        )
        plt.axis("off")
