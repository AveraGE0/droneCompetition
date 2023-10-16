"""Main module to provide evaluation functions"""
import numpy as np
from abc import ABC


class ResultFormatter(ABC):
    """Interface for generation result scores (used for fitness)"""
    def foarmat(self, **kwargs) -> tuple:
        """Interface method. Must return a tuple of scores for an individual

        Returns:
            tuple: results, should be floats
        """
        raise NotImplementedError


class DronePathResultFormatter(ResultFormatter):
    """Class to get results for a drone path given the segmented path"""
    
    def format(self, **kwargs) -> tuple:
        """Function to get results for the actual path a drone took compared to the segments that
        where available

        Args:
            actual_path (list): list of indices of the path the drone took
            segment_indices (np.ndarray): indices of the segments

        Returns:
            tuple: scores of the simulation - (finished, ratio_off_line, completion_time)
        """
        actual_path: list = kwargs.get("actual_path")
        segment_indices: np.ndarray = kwargs.get("segmented_indices")
        finish_area: tuple = kwargs.get("finish_area")
        on_path = 0
        total_steps = 0
        finished = False
        for coords in actual_path:
            total_steps += 1
            if tuple(coords) in segment_indices:
                on_path+= 1
            if finish_area[0] < coords[0] < finish_area[2] and finish_area[1] < coords[1] < finish_area[3]:
                finished = True
                break
        return finished, on_path / total_steps, total_steps
