import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio
from IPython.display import Image
import numpy as np
import os

root = os.getcwd()
from pathlib import Path


def plot_image(data, title="Image", file_path=None):
    fig = plt.figure()
    plt.imshow(data, cmap="gray")
    plt.title(title)
    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
    plt.show()


def histogram_image(data, title="Histogram of image", nb_bins=256, file_path=None):
    """
    :param data:
    :param title:
    :param nb_bins:
    :param file_path:
    :return:
    """
    plt.figure()
    plt.hist(data.flatten(), bins=nb_bins)
    plt.title(title)
    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
    plt.show()


def plot_histogram_threshold(data, lst_threshold, title="Histogram of image + thresh", nb_bins=256, file_path=None):
    """
    :param data:
    :param lst_threshold:
    :param title:
    :param nb_bins:
    :param file_path:
    :return:
    """
    plt.figure()
    plt.hist(data.flatten(), bins=nb_bins)
    for k in lst_threshold:
        plt.axvline(k, color='r')
    plt.title(title)
    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
    plt.show()


def plot_series_slice(data, fig_rows, fig_cols, dimension=0, rot=False, title="Serie of slices", file_path=None):
    """
    :param data:
    :param fig_rows:
    :param fig_cols:
    :param dimension:
    :param rot:
    :param title:
    :param file_path:
    :return:
    """
    n_subplots = fig_rows * fig_cols
    n_slice = data.shape[dimension]
    step_size = n_slice // n_subplots
    plot_range = n_subplots * step_size
    start_stop = int((n_slice - plot_range) / 2)

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[15, 20], constrained_layout=True)

    for idx, img in enumerate(range(start_stop, plot_range, step_size)):
        if dimension == 0:
            image = data[img, :, :]
        elif dimension == 1:
            image = data[:, img, :]
        else:
            image = data[:, :, img]
        if rot:
            image = np.rot90(image)
        else:
            image = image
        axs.flat[idx].imshow(image, cmap='gray')
        axs.flat[idx].axis('off')
        axs.flat[idx].set_title("Slice, %s." % img)

    fig.suptitle(title, fontsize=16)
    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
    plt.show()
    return


def plot_brain_MRI(data, figsize, lst_slice=None, lst_names=None, lst_rot=None, file_path=None):
    """
    :param data:
    :param figsize:
    :param lst_slice:
    :param lst_names:
    :param lst_rot:
    :param file_path:
    :return:
    """
    if lst_slice is None:
        x, y, z = np.shape(data)
        lst_slice = [np.random.randint(0, x), np.random.randint(0, y), np.random.randint(0, z)]
    if lst_names is None:
        lst_names = ["figure 1", "figure 2", "figure 3"]
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    fig.set_size_inches(figsize[0], figsize[1])
    if lst_rot is None:
        lst_rot = [False, False, False]

    if lst_rot[0]:
        im1 = np.rot90(data[lst_slice[0], :, :])
    else:
        im1 = data[lst_slice[0], :, :]
    axs[0].imshow(im1, cmap="gray")
    axs[0].title.set_text(lst_names[0])
    axs[0].title.set_size(fontsize=20)
    axs[0].axis('off')

    if lst_rot[1]:
        im2 = np.rot90(data[:, lst_slice[1], :])
    else:
        im2 = data[:, lst_slice[1], :]
    axs[1].imshow(im2, cmap="gray")
    axs[1].title.set_text(lst_names[1])
    axs[1].title.set_size(fontsize=20)
    axs[1].axis('off')

    if lst_rot[2]:
        im3 = np.rot90(data[:, :, lst_slice[2]])
    else:
        im3 = data[:, :, lst_slice[2]]
    axs[2].imshow(im3, cmap="gray")
    axs[2].title.set_text(lst_names[2])
    axs[2].title.set_size(fontsize=20)
    axs[2].axis('off')

    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
    plt.show()

    return


def plot_freq_image(data, title="Image in Frequency domain", file_path=None):
    plot_image(np.log(np.abs(data)), title, file_path)


def showSeedsPos(seedPositionsList, imagesList, file_path=None):
    """
    :param seedPositionsList:
    :param imagesList:
    :param file_path:
    :return:
    """
    plt.figure(figsize=(16, 6))

    for imageIndex, image in enumerate(imagesList):
        plt.subplot(1, len(imagesList), imageIndex + 1)
        plt.title('Seed : ' + str(seedPositionsList[imageIndex][0]) + ' - ' + str(seedPositionsList[imageIndex][1]))
        plt.imshow(image, cmap='gray')
        plt.plot(seedPositionsList[imageIndex][1], seedPositionsList[imageIndex][0], color='r', marker="+",
                 markersize=10)
    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)

    plt.show()


def evaluate_ROI(data, roi, fig_rows, fig_cols, dimension=0, rot=False, title="Serie of slices + ROI", file_path=None):
    n_subplots = fig_rows * fig_cols
    n_slice = data.shape[dimension]
    step_size = n_slice // n_subplots
    plot_range = n_subplots * step_size
    start_stop = int((n_slice - plot_range) / 2)

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[15, 20], constrained_layout=True)

    for idx, img in enumerate(range(start_stop, plot_range, step_size)):
        if dimension == 0:
            image = data[img, :, :]
            roi_2D = roi[1:]
        elif dimension == 1:
            image = data[:, img, :]
            roi_2D = [roi[0], roi[2]]
        else:
            image = data[:, :, img]
            roi_2D = roi[:2]
        if rot:
            image = np.rot90(image)
        else:
            image = image
        rect = mpatches.Rectangle((roi_2D[1][0], roi_2D[0][0]),
                                  roi_2D[1][1] - roi_2D[1][0], roi_2D[0][1] - roi_2D[0][0],
                                  fill=False, edgecolor='red', linewidth=2)

        axs.flat[idx].imshow(image, cmap="gray")
        axs.flat[idx].add_patch(rect)
        axs.flat[idx].axis('off')
        axs.flat[idx].set_title("Slice, %s." % img)

    fig.suptitle(title, fontsize=16)
    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
    plt.show()
    return


def visualize_data_gif(data, file_path, rot=False):
    """
    :param data:
    :param file_path:
    :param rot:
    :return:
    """
    images = []
    for i in range(data.shape[0]):
        x = data[min(i, data.shape[0] - 1), :, :]
        if rot:
            x = np.rot90(x)
        y = data[:, min(i, data.shape[1] - 1), :]
        if rot:
            y = np.rot90(y)
        z = data[:, :, min(i, data.shape[2] - 1)]
        if rot:
            z = np.rot90(z)
        # print(np.shape(x),np.shape(y),np.shape(z))
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)

    imageio.mimsave(file_path, images, duration=0.05)
    return Image(filename=file_path, format='png')


def plot_region_growing(img, region, title="Result of region growing", file_path=None):
    """
    :param img:
    :param region:
    :param title:
    :param file_path:
    :return:
    """
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    fig.set_size_inches(16, 5)
    fig.suptitle(title, fontsize=20)

    axes[0].imshow(region, cmap='gray')
    axes[0].set_title('Growed region', fontsize=18)
    axes[1].imshow(img, cmap='gray')
    axes[1].set_title('Original Image', fontsize=18)
    axes[2].imshow(region, cmap='gray')
    axes[2].imshow(img, cmap='gray', alpha=0.5)
    axes[2].set_title('Superposition of the 2 images', fontsize=18)

    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
    plt.show()
    return


def plot_original_noise(original_img, noised_img, title="Comparision between original image and noised image",
                        file_path=None):
    """
    :param original_img:
    :param noised_img:
    :param title:
    :param file_path:
    :return:
    """

    diff = noised_img - original_img
    diff[diff < 0] = 0
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(title, fontsize=25)

    axs[0].imshow(original_img, cmap="gray")
    axs[0].title.set_text("Image without noise")
    axs[0].title.set_size(fontsize=20)

    axs[1].imshow(noised_img, cmap="gray")
    axs[1].title.set_text("Image with noise")
    axs[1].title.set_size(fontsize=20)

    axs[2].imshow(diff, cmap="gray")
    axs[2].title.set_text("Noise")
    axs[2].title.set_size(fontsize=20)
    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
    plt.show()
    return


def polt_pourcentage(data, title="Bar plot of pourcentage", file_path=None):
    x = list(data.keys())
    y = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(x, y, color='maroon',
            width=0.4)

    plt.title(title)
    if file_path is not None:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)
    plt.show()
