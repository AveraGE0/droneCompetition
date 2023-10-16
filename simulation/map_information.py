"""Main module to present an interface that presents information about maps"""
from dataclasses import dataclass


@dataclass
class MapInformation:
    """Class to represent information about the available maps"""
    shared_map_name: str
    map_size: tuple[int, int]
    map_amount: int
    start_points: list[tuple[int, int]]
    end_areas: list[tuple[int, int, int, int]]
