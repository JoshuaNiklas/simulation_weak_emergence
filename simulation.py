from multiprocessing import Manager, Process
import matplotlib.pyplot as plt
import numpy as np
import random
from agent import Agent


class Simulation:

    def __init__(self, simulation_size, simulation_speed, n_agents):
        self.simulation_size = simulation_size
        self.dt = simulation_speed
        self.n_agents = n_agents
        self.manager = Manager()
        self.shared_memory = self.manager.dict()
        self.agents = []

        #Create and add Agents
        for i in range(n_agents):
            starting_position = self.create_starting_position()
            colors = self.create_random_colors()
            agent = Agent(i, starting_position, colors[i], self.shared_memory)
            self.agents.append(agent)

    def create_starting_position(self):
        starting_position = [list(np.random.uniform(0, self.simulation_size, 2)) for _ in range(2)]
        starting_position = self.adjust_points(starting_position)
        return starting_position 
        
    #Makes sure that the first point given by the position is the lower left point whereas the second one is the higher right one
    def adjust_points(self, positions):
        if positions[0][0] > positions[1][0]:
            positions[0][0], positions[1][0] = positions[1][0].copy(), positions[0][0].copy()

        if positions[0][1] > positions[1][1]:
            positions[0][1], positions[1][1] = positions[1][1].copy(), positions[0][1].copy()
        return positions

    def create_random_colors(self):
        colors = [(random.random(), random.random(), random.random()) for _ in range(self.n_agents)]
        return colors

    def run(self, timestep):
        pass