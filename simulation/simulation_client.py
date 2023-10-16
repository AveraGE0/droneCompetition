"""
A client is responsible for the actual simulation. This only includes simulating the behavior
of the drone (very simple), which is derived from a given decision tree. Therefore a client
is initiated with a decision tree (brain) and a reference to the map manager (shared memory).
"""
from pathlib import Path
import pandas as pd
from typing import Callable
from simulation.brain import Brain
from simulation.map_information import MapInformation
from simulation.simulation_parameter import SimulationConfig
from simulation.image_processing import StartDistProcessing, ImageProcessing
from simulation.result_functions import ResultFunction, DronePathEvaluation
from multiprocessing.shared_memory import SharedMemory
import numpy as np
# only for debug
import matplotlib.pyplot as plt
from math import sqrt


def simulate_drone(
        brain: Brain,
        map_info: MapInformation,
        sim_info: SimulationConfig,
        image_processing: ImageProcessing,
        eval_func: ResultFunction
    ) -> dict:
    """Simulation function to actually simulate a brain (decision maker) on different environments
    (which can be accessed over the shared memory)

    Args:
        brain (Brain): Gets inputs depending on the current position on the map
        map_info (MapInformation): Object containing information about the map
        sim_info (SimulationConfig): Object containing information about the simulation
        image_processing (ImageProcessing): Object handles the image processing

    Returns:
        dict: scores of the simulation - depending on the eval function
    """
    map_provider = SharedMemory(map_info.shared_map_name, create=False)
    maps = np.ndarray(shape=list(map_info.map_size) + [map_info.map_amount], dtype=np.float32, buffer=map_provider.buf)
    # simulation
    scores = []
    for i_map in range(maps.shape[2]):
        position = list(map_info.start_points[i_map])
        positions = []
        map_path_coords = list(zip(*np.where(maps[:,:,i_map] > 0.5)))

        debug_img = maps[:,:,i_map].copy()
        end_area = map_info.end_areas[i_map]
        for step in range(sim_info.max_sim_steps):
            positions.append(position.copy())  # add current position
            debug_img[position[0], position[1]] = 2  # TODO: remove

            current_view = np.zeros(shape=(128, 128))
            # set the area of current view that is overridden (in case we are to close to the map border)
            min_x, max_x = abs(min(position[0] - 64, 0)), 128 - max(0, (position[0]+64 - 1000))
            min_y, max_y = abs(min(position[1] - 64, 0)), 128 - max(0, (position[1]+64 - 1000))
            
            sel_min_x, sel_max_x = max(position[0] - 64, 0), min(position[0] + 64, map_info.map_size[0])
            sel_min_y, sel_max_y = max(position[1] - 64, 0), min(position[1] + 64, map_info.map_size[1])
            
            current_view[min_x:max_x, min_y:max_y] = maps[sel_min_x:sel_max_x, sel_min_y:sel_max_y, i_map]
            # check bounds
            if not (0 < position[0] < map_info.map_size[0] and 0 < position[1] < map_info.map_size[1]):
                print("out of bounds, aborting")
                break  # out of bounds!
            # check finish area
            if (end_area[0] < position[0] < end_area[2] and end_area[1] < position[1] < end_area[3]):
                print(f"in finish area, stopping: {position}")
                break  # we are in the finish area!
            # get line distances to path edge
            img_results = image_processing.process_image(current_view)
            # check if we are too far from the track
            if sum(img_results) == 0:
                print("off track, aborting")
                break
            # update position
            direction = brain.process(img_results)
            norm = max(sqrt(direction[0]**2 + direction[1]**2), 1)
            position[0] += int(direction[0] // norm) * 2
            position[1] += int(direction[1] // norm) * 2
            if len(positions) >= 5 and position[0] == positions[4][0] and position[1] == positions[4][1]:
                print("That bitch aint moving")
                break
            plt.imshow(current_view)
            plt.show()
        if step == sim_info.max_sim_steps - 1:
            print("Simulation step timeout.")
        # get evaluation scores (for the fitness function)
        scores.append(eval_func.evaluate(actual_path=positions, segmented_indices=map_path_coords, finish_area=map_info.end_areas[i_map]))
        plt.imshow(debug_img)
        plt.show()
        #plt.imshow(current_view)
        #plt.show()
        # TODO: write positions to file for testing
    return scores
