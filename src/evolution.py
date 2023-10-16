import operator
import random
import math
import pickle
import numpy as np
from pathlib import Path
# deap stuff
from deap import base, creator, tools
import deap.gp as gp
import deap.algorithms as algorithms
# import custom stuff
from simulation.simulation_manager import SharedMemory
from simulation.simulation_client import simulate_drone
from simulation.brain import ParseTreeBrain
from simulation.result_functions import DronePathResultFormatter
from simulation.image_processing import StartDistProcessing
from simulation.map_information import MapInformation
from simulation.simulation_parameter import SimulationConfig


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def create_pset():
    pset = gp.PrimitiveSet("Controller", 8)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(min, 2)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.abs, 1)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addEphemeralConstant("rdom", lambda: random.uniform(-1, 1))

    pset.renameArguments(ARG0="N")
    pset.renameArguments(ARG1="NE")
    pset.renameArguments(ARG0="E")
    pset.renameArguments(ARG1="SE")
    pset.renameArguments(ARG0="S")
    pset.renameArguments(ARG1="SW")
    pset.renameArguments(ARG0="W")
    pset.renameArguments(ARG1="NW")
    return pset


def run_evolution() -> None:
    # create simulation specific stuff
    result_formatter = DronePathResultFormatter()
    img_proc = StartDistProcessing()
    name = "test_tracks"
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
        # load maps
        maps = np.load(Path("tracks/test_tracks.npy"))
        for i in range(maps.shape[2]):
            maps[:, : , i] = maps[:, :, i].T
    # Parameter setup
    POP_SIZE = 10
    GENERATIONS = 10
    # setup evolution
    pset = create_pset()
    creator.create("FitnessMax", base.Fitness, weights=(10, 5))
    creator.create("DroneIndividual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.DroneIndividual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, POP_SIZE)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("get_best", tools.selBest, k=1)

    def eval_function(individuals: list) -> tuple:
        # tree -> function
        individual_xy, = individuals
        controller_xy = toolbox.compile(expr=individual_xy)
        # run simulation
        scores = simulate_drone(
            ParseTreeBrain(controller_xy),
            map_info=map_info,
            sim_info=sim_info,
            image_processing=img_proc,
            result_formatter=result_formatter
        )
        return (
            sum([1 for score in scores if score[0]]),
            np.mean([score[1] for score in scores])
        ) 
    
    toolbox.register("evaluate", eval_function)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=10)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # main simulation loop
    try:
        # create shared memory
        sm = SharedMemory(name=name, create=True, size=maps.nbytes)
        # load maps into shared memory
        maps_shared = np.ndarray(shape=maps.shape, dtype=np.float32, buffer=sm.buf)
        maps_shared[:, :, :] = maps[:, :, :]

        # Start simulation here
        random.seed(318)
        # use coevolution, controller contains a species for both x and y (two populations)
        controller_pop = toolbox.population()
        g = 0
        while g < GENERATIONS:
            controller_pop = algorithms.varAnd(controller_pop, toolbox, 0.6, 1.0)
            for controller in controller_pop:
                # Evaluate and set the individual fitness
                controller.fitness.values = toolbox.evaluate([controller])
            # Select the individuals
            controller_pops = toolbox.select(controller_pop, len(controller_pop))  # Tournament selection
        pop = toolbox.population(n=300)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                    halloffame=hof, verbose=True)
    finally:
        #sm.close()
        sm.unlink()
        # print log
    return pop, log, hof



if __name__ == "__main__":
    pop, log, hof = run_evolution()
