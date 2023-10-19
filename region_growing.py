import numpy as np


def neighbours_not_checked(xPixPos=None, yPixPos=None,zPixPos=None, checkedImage=None, vect=None):
    for i in range(xPixPos - 1, xPixPos + 2):
        for j in range(yPixPos - 1, yPixPos + 2):
            for k in range(zPixPos-1,zPixPos+2):
                if checkedImage[i, j,k] != 1:
                    checkedImage[i, j,k] = 1
                    vect.append([i, j,k])


def region_growing(seedPosition=None, image=None, rangeIntensity=None):
    region_growed = np.zeros_like(image)
    checked_image = np.zeros_like(image)
    xpp, ypp, zpp = seedPosition[0], seedPosition[1], seedPosition[2]
    checked_image[xpp, ypp, zpp] = 1
    region_growed[xpp, ypp, zpp] = 1
    intensities = [image[xpp, ypp, zpp]]
    vector2verif = []
    neighbours_not_checked(xPixPos=xpp, yPixPos=ypp,zPixPos=zpp, checkedImage=checked_image, vect=vector2verif)

    while len(vector2verif) > 0:
        inten_mean = np.mean(intensities)
        pixel = vector2verif[0]
        intensity = image[pixel[0]][pixel[1]][pixel[2]]
        if (intensity - rangeIntensity < inten_mean) and (intensity + rangeIntensity > inten_mean):
            region_growed[pixel[0], pixel[1],pixel[2]] = 1
            neighbours_not_checked(pixel[0], pixel[1],pixel[2], checked_image, vector2verif)
            intensities.append(image[pixel[0], pixel[1],pixel[2]])
        vector2verif = vector2verif[1:]
    return region_growed, checked_image


seed_pixel = seedPosList[0][:]
growed_givenImage, check_givenImage = region_growing(seedPosition=seed_pixel, image=imglFilt, rangeIntensity=40)