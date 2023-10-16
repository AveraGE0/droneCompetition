"""Dataclass to represent the parameters set for a simulation run."""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Class for storing run specific parameters"""
    name: str  # name of the run
    max_sim_steps: int  # max iterations for the drone to run
    view_size: tuple[int, int]  # view size of the drone, should be 128x128
    out_path: Path  # path to a file for the history of the drone positions