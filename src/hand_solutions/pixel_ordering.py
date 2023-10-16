import cv2
import numpy as np
from math import sqrt
import numpy as np


def order_pixels(pixel_coords: list, image_width) -> tuple:
    """Function that receives an unordered list of pixel coordinates and returns
    a tuple with the same elements in correct order

    Args:
        pixel_coords (list): list of tuples (x, y)

    Returns:
        tuple: ordered coords
    """
    #pixel_coords = np.asarray(pixel_coords)
    ordered: list = []
    # make dist matrix
    dist_matrix = np.zeros(shape=(len(pixel_coords), len(pixel_coords)))
    for i1, pix in enumerate(pixel_coords):
        for i2, pix2 in enumerate(pixel_coords):
            dist_matrix[i1, i2] = np.linalg.norm(pix-pix2)  # euclidean dist

    # in matrix, start at point closest to 0 
    dists = [np.linalg.norm(dist) for dist in pixel_coords - np.array([image_width/2, image_width/2])]
    current_point = np.argmin(dists)
    while(len(ordered) != len(pixel_coords)):
        # save index of current value in ordered if
        ordered.append(current_point)
        # go to row of lowest value in the current row, excluding already found ones
        index_of_min = None
        for i in range(dist_matrix.shape[0]):
            if i in ordered:
                continue
            if not index_of_min:
                index_of_min = i
            if dist_matrix[current_point, i] < dist_matrix[current_point, index_of_min]:
                index_of_min = i
        current_point = index_of_min
    return pixel_coords[ordered]