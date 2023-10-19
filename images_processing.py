import random
import numpy as np
import copy


def add_noise_salt_and_peper(img):
    """
    :param img:
    :return:
    """
    # Getting the dimensions of the image
    row, col = img.shape
    row_col = row * col
    # Randomly pick some pixels in the
    # image for coloring them white
    number_of_pixels = random.randint(row_col // 8, row_col // 6)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(row_col // 8, row_col // 6)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


def add_noise_uniform(img, mu=0, sigma=1):
    """
    :param img:
    :param mu:
    :param sigma:
    :return:
    """
    noise = np.random.normal(mu, sigma, size=img.shape)
    return img + noise


def post_region_growing(data):
    new_data = copy.deepcopy(data)
    m, n, l = np.shape(data)
    for i in range(m):
        for j in range(n):
            for k in range(l):
                if data[i, j, k] == 1:
                    nb = 0
                    if 0 <= i - 1 and data[i - 1, j, k] == 1:
                        nb += 1
                    if m > i + 1 and data[i + 1, j, k] == 1:
                        nb += 1
                    if 0 <= j - 1 and data[i, j - 1, k] == 1:
                        nb += 1
                    if n >= j + 1 and data[i, j + 1, k] == 1:
                        nb += 1
                    if 0 <= k - 1 and data[i, j, k - 1] == 1:
                        nb += 1
                    if l >= k + 1 and data[i, j, k + 1] == 1:
                        nb += 1

                    if nb < 3:
                        new_data[i, j, k] = 0
                elif data[i, j, k] == 0:
                    nb = 0
                    if 0 <= i - 1 and data[i - 1, j, k] == 1:
                        nb += 1
                    if m > i + 1 and data[i + 1, j, k] == 1:
                        nb += 1
                    if 0 <= j - 1 and data[i, j - 1, k] == 1:
                        nb += 1
                    if n >= j + 1 and data[i, j + 1, k] == 1:
                        nb += 1
                    if 0 <= k - 1 and data[i, j, k - 1] == 1:
                        nb += 1
                    if l >= k + 1 and data[i, j, k + 1] == 1:
                        nb += 1

                    if nb > 5:
                        new_data[i, j, k] = 1
    return new_data


def cross_sectional(data,dimension,nb_slices):
    n_slice = data.shape[dimension]
    step_size = n_slice // nb_slices
    result = {}
    for k in range(0,n_slice,step_size):
        if dimension == 0:
            pourcentage = np.count_nonzero(data[k,:,:])/(data.shape[1]*data.shape[2])
        elif dimension == 1:
            pourcentage = np.count_nonzero(data[:, k, :]) / (data.shape[0] * data.shape[2])
        elif dimension == 2:
            pourcentage = np.count_nonzero(data[:, :, k]) / (data.shape[1] * data.shape[0])
        else:
            print("Are you stupid?")
            pourcentage = 0
        result[k] = pourcentage

    return result
