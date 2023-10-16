"""Module to represent the image processing that is applied during the runs"""
from abc import ABC
import numpy as np
from math import sqrt


class ImageProcessing(ABC):
    """Main interface for the image processing"""
    def __init__(self) -> None:
        super().__init__()
    
    def process_image(self, view: np.ndarray) -> tuple:
        raise NotImplementedError


class StartDistProcessing(ImageProcessing):
    """Class to represent the process of ray tracing, where rays are
    sent out in a star patter, and the intersection of those rays with
    the track is measured and returned."""
    def process_image(self, view: np.ndarray, lines=8) -> tuple:
        """Method to get the distance of lines to the center of the drone.

        Args:
            view (np.ndarray): the current view of the drone, default should be 128x128

        Returns:
            tuple: 8 floats with distances, representing distances to N, NE, E, ES, S, SW, W, NW
        """
        center_x = view.shape[0] // 2
        center_y = view.shape[1] // 2
        angles = [0 + 45 * pos for pos in range(lines)]
        radius = 20  # dont need to search for more than 20px
        intersections: list[float] = []
        for angle in angles:
            intersect_x, intersect_y = self.find_line_intersection(center_x, center_y, angle, radius, view)
            intersections.append(sqrt((intersect_x-center_x)**2 + (intersect_y-center_y)**2))
        return tuple(intersections)

        
    def find_line_intersection(self, center_x, center_y, angle_degrees, radius, view):
        """Either returns the point of intersection with a line or returns the radius.
        This function was mainly generated by chatgpt but modified

        Args:
            center_x (int): x
            center_y (int): y
            angle_degrees (float): angle for line
            radius (int): max length for line
            view (np.ndarray): view (to see intersection)

        Returns:
            _type_: _description_
        """
        # Convert angle from degrees to radians
        angle_radians = np.deg2rad(angle_degrees)
        
        # Calculate the endpoint coordinates
        end_x = center_x + radius * np.cos(angle_radians)
        end_y = center_y + radius * np.sin(angle_radians)
        
        # Generate pixel coordinates along the line
        num_points = int(radius)  # You can adjust the number of points
        
        # previous value will be the last detected value if there is one
        prev_x, prev_y = center_x, center_y
        x, y = center_x, center_y

        for i in range(1, num_points + 1):
            t = i / num_points
            next_x = int((1 - t) * center_x + t * end_x)
            next_y = int((1 - t) * center_y + t * end_y)
            if view[x, y] != view[next_x, next_y]:
                if view[next_x, next_y] == 1:
                    return next_x, next_y
                else:
                    return prev_x, prev_y
            prev_x, prev_y = next_x, next_y
        # if no edge was found return the last position if we only saw ones
        if view[prev_x, prev_y] == 1:
            x, y = prev_x, prev_y
        return (x, y)

