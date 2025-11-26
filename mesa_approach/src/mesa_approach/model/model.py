"""
Optimal Rectangle Packing Model
===================
A Mesa implementation of Optimal Rectangle Packing Problem as proposed by Richard E. Korf, e.g. https://arxiv.org/pdf/1402.0557
Uses Mesa as a means for Multi-Agent-System
"""

import os
import sys

#Get the fourth-top most directory and resolve from there
sys.path.insert(0, os.path.abspath("../../../.."))


import numpy as np

from mesa import Model
from mesa_approach.agents.agents import Rectangle
from mesa.experimental.continuous_space import ContinuousSpace


class OptimalRectanglePackingModel(Model):
    """Optimal Rectangle Packing Model class. Handles agent creation, placement and scheduling."""

    def __init__(
        self,
        population_size=5,
        width=100,
        height=100,
        speed=1,
        vision=10,
        separation=2,
        cohere=0.03,
        separate=0.015,
        match=0.05,
        seed=None,
    ):
        """Create a new Optimal Rectangle Packing Model.

        Args:
            population_size: Number of Rectangles in the simulation (default: 5)
            width: Width of the space (default: 100)
            height: Height of the space (default: 100)
            speed: How fast the Boids move (default: 1)
            vision: How far each Boid can see (default: 10)
            separation: Minimum distance between Boids (default: 2)
            cohere: Weight of cohesion behavior (default: 0.03)
            separate: Weight of separation behavior (default: 0.015)
            match: Weight of alignment behavior (default: 0.05)
            seed: Random seed for reproducibility (default: None)
        """
        super().__init__(seed=seed)
        self.agent_angles = np.zeros(
            population_size
        )  # holds the angle representing the direction of all agents at a given step

        # Set up the space
        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=False,
            random=self.random,
            n_agents=population_size,
        )

        # Create and place the Boid agents
        

        for i in range(population_size):
            rectangle = self.spawn_rectangle()
            rectangle.speed = speed
            self.agents.add(rectangle)

    def step(self):
        """Run one step of the model.

        All agents are activated in random order using the AgentSet shuffle_do method.
        """
        self.agents.shuffle_do("step")

    def check_overlap(self, rect1, rect2):
        """Check if two rectangles overlap."""
        x1_min = rect1.position[0] - rect1.width / 2
        x1_max = rect1.position[0] + rect1.width / 2
        y1_min = rect1.position[1] - rect1.height / 2
        y1_max = rect1.position[1] + rect1.height / 2

        x2_min = rect2.position[0] - rect2.width / 2
        x2_max = rect2.position[0] + rect2.width / 2
        y2_min = rect2.position[1] - rect2.height / 2
        y2_max = rect2.position[1] + rect2.height / 2

        return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)
    
    def spawn_rectangle(self):
        while True:
            position = self.rng.random(size=2) * np.array([self.space.x_max, self.space.y_max])
            rect_height = self.rng.integers(5, 20)
            rect_width = self.rng.integers(5, 20)
            new_rect = Rectangle(self, self.space, position=position, height=rect_height, width=rect_width)
            overlap = any(self.check_overlap(new_rect, agent) for agent in self.agents if new_rect != agent)
            if not overlap:
                return new_rect
