"""
Simulation Manager module. The simulation manager is responsible for creating the subprocesses
that run the actual simulation. Also the map manager, that shares maps across all processes
is setup and started by this process.
"""
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool
import numpy as np
from simulation.brain import Brain

class SimulationManager:
    def __init__(self, simulation_clients: int) -> None:
        # initialize shared memory (maps)
        maps = np.ndarray(shape=(1000, 1000, 5))
        self.maps = SharedMemory(name="MapDistributer", size=maps.nbytes)
        maps = np.ndarray(maps.shape, dtype=np.float32, buffer=self.maps.buf)
        # TODO: load maps here
        self.simulation_pool = Pool(processes=3)
        self.simulation_clients = simulation_clients

    def simulate(self, brains: list[Brain]) -> list[dict[str, float]]:
        """Main interface method for simulation. Receives a list of brain, which
        are classes that map an input to a response. The simulation distributes
        the brains across clients and simulates the behavior of the drones in parallel.

        Args:
            brains (list[Brain]): A list of brains (phenotypes of individuals) to be evaluated

        Returns:
            list[float]: Simulation scores as dictionaries
        """
        # TODO: insert correct function, arguments
        simulation_scores = [self.simulation_pool.apply_async() for client in range(self.simulation_clients)]
        self.simulation_pool.close()
        self.simulation_pool.join()
        return [res.get() for res in simulation_scores]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for client in self.simulation_clients:
            # TODO: destroy clients
            pass
        self.maps.unlink()
