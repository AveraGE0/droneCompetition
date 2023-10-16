import pytest
from simulation.simulation_client import simulate_drone
from simulation.brain import DecisionTreeBrain, NonTerminalNode, TerminalNode
from simulation.result_functions import DronePathEvaluation
from simulation.image_processing import StartDistProcessing
from simulation.map_information import MapInformation
from simulation.simulation_parameter import SimulationConfig
from simulation.simulation_manager import SharedMemory
import pickle
import numpy as np
from pathlib import Path


def make_brain() -> DecisionTreeBrain:
    return DecisionTreeBrain(
        NonTerminalNode(
            NonTerminalNode(
                TerminalNode((1, 0)),
                TerminalNode((1, 1)),
                lambda x: x[7] < x[0]
            ),
            NonTerminalNode(
                TerminalNode((1, 0)),
                TerminalNode((-1, 1)),
                lambda x: x[1] < x[0]
            ),
            lambda x: x[7] > x[1]
        )
    )

def test_map_cropping():
    map_info = MapInformation(
        shared_map_name="",
        map_size=(1000, 1000),
        map_amount=3,
        start_points=[],
        end_areas=[]
    )
    positions = [(10, 10), (200, 200), (950, 950), (950, 300), (300 , 950)]
    for position in positions:
        print(position)
        min_x, max_x = abs(min(position[0] - 64, 0)), 128 - max(0, (position[0]+64 - 1000))
        min_y, max_y = abs(min(position[1] - 64, 0)), 128 - max(0, (position[1]+64 - 1000))
        print(f"{min_x}:{max_x},{min_y}:{max_y}")
        sel_min_x, sel_max_x = max(position[0] - 64, 0), min(position[0] + 64, map_info.map_size[0])
        sel_min_y, sel_max_y = max(position[1] - 64, 0), min(position[1] + 64, map_info.map_size[1])
        print(f"{sel_min_x}:{sel_max_x}, {sel_min_y}:{sel_max_y}")
        print(f"{min_x + sel_max_x- sel_min_x}={max_x}, {min_y + sel_max_y - sel_min_y}={max_y}")

def test_sim():
    # load brain
    name = "test_tracks"
    # load map info
    img_proc = StartDistProcessing()
    eval_func = DronePathEvaluation()
    with open("tracks/test_tracks_info.pickle", 'rb') as in_file:
        info_dict = pickle.load(in_file)
        map_info = MapInformation(
            shared_map_name=name,
            map_size=info_dict["maps_shape"][0:2],
            map_amount=info_dict['maps_shape'][2],
            start_points=info_dict["start_points"],
            end_areas=info_dict["end_areas"]
        )
        sim_info = SimulationConfig("test1", 1000, (128, 128), "sim_runs/test1")
        # load map
        maps = np.load(Path("tracks/test_tracks.npy"))
        for i in range(maps.shape[2]):
            maps[:, : , i] = maps[:, :, i].T
        try:
            sm = SharedMemory(name=name, create=True, size=maps.nbytes)
            maps_shared = np.ndarray(shape=maps.shape, dtype=np.float32, buffer=sm.buf)
            maps_shared[:, :, :] = maps[:, :, :]
            brain = make_brain()
            scores = simulate_drone(brain, map_info, sim_info,img_proc, eval_func)
            fitness = sum([1 for score in scores if score[0]]) + 1 * np.mean([score[1] for score in scores])
            print(f"Drone finished {sum([1 for completed in scores if completed[0]])}/{len(scores)} maps")
            print(f"Average {np.mean([score[1] for score in scores])}% staying on track.")
            print(f"Fitness: {fitness}")
        finally:
            #sm.close()
            sm.unlink()

if __name__ == '__main__':
    test_sim()
