import pytest
from simulation.image_processing import StartDistProcessing
import numpy as np
import matplotlib.pyplot as plt


def test_line_coordinates():
    # Example usage:
    center_x = 50
    center_y = 50
    angle_degrees = 90  # Angle in degrees
    radius = 30

    line_coordinates = StartDistProcessing().generate_line_coordinates(center_x, center_y, angle_degrees, radius)

    test_image = np.zeros(shape=(100, 100))
    for coords in line_coordinates:
        test_image[coords[0], coords[1]] = 1
    plt.imshow(test_image)
    plt.color_bar()
    plt.show()


if __name__ == '__main__':
    test_line_coordinates()
