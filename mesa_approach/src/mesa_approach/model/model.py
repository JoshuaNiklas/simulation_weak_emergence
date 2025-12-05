"""
Optimal Rectangle Packing Model
===================
A Mesa implementation of Optimal Rectangle Packing Problem as proposed by Richard E. Korf, e.g. https://arxiv.org/pdf/1402.0557
Uses Mesa as a means for Multi-Agent-System
"""

import os
import sys

# Get the fourth-top most directory and resolve from there
sys.path.insert(0, os.path.abspath("../../../.."))

import numpy as np
from intervaltree import IntervalTree

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
            speed: How fast the rectangles move (default: 1)
            vision: How far each rectangle can see (default: 10)
            separation: Minimum distance between rectangles (default: 2)
            cohere: Weight of cohesion behavior (default: 0.03)
            separate: Weight of separation behavior (default: 0.015)
            match: Weight of alignment behavior (default: 0.05)
            seed: Random seed for reproducibility (default: None)
        """
        super().__init__(seed=seed)

        self.rng = np.random.default_rng(seed)

        self.space_width = float(width)
        self.space_height = float(height)

        self.x_tree = IntervalTree()
        self.y_tree = IntervalTree()

        self.agent_list = []

        self.space = ContinuousSpace(
            [[0, self.space_width], [0, self.space_height]],
            torus=False,
            random=self.random,
            n_agents=population_size,
        )

        # Create and place the Rectangle agents (sizes scaled to world)
        MIN_FRAC = 0.02
        MAX_FRAC = 0.18

        for _ in range(population_size):
            rect_h = float(self.rng.uniform(MIN_FRAC, MAX_FRAC) * self.space_height)
            rect_w = float(self.rng.uniform(MIN_FRAC, MAX_FRAC) * self.space_width)
            rect = self.spawn_rectangle(rect_h, rect_w)
            rect.speed = speed
            self.agent_list.append(rect)
            self.add_agent_intervals(rect)

    def step(self):
        """Advance the model by one step - step all agents."""
        for agent in self.agent_list:
            agent.step()

    def _rect_interval(self, position, width, height):
        """Return axis intervals (x_min, x_max, y_min, y_max)."""
        x_min = float(position[0] - width / 2.0)
        x_max = float(position[0] + width / 2.0)
        y_min = float(position[1] - height / 2.0)
        y_max = float(position[1] + height / 2.0)
        return x_min, x_max, y_min, y_max

    def add_agent_intervals(self, agent):
        x_min, x_max, y_min, y_max = self._rect_interval(agent.position, agent.width, agent.height)
        self.x_tree.addi(x_min, x_max, agent)
        self.y_tree.addi(y_min, y_max, agent)

    def remove_agent_intervals(self, agent):
        to_remove_x = [iv for iv in list(self.x_tree) if iv.data is agent]
        for iv in to_remove_x:
            self.x_tree.remove(iv)
        to_remove_y = [iv for iv in list(self.y_tree) if iv.data is agent]
        for iv in to_remove_y:
            self.y_tree.remove(iv)

    def update_agent_intervals(self, agent, old_position=None):
        self.remove_agent_intervals(agent)
        self.add_agent_intervals(agent)

    def potential_overlaps_for_rect(self, position, width, height):
        x_min, x_max, y_min, y_max = self._rect_interval(position, width, height)
        x_hits = set(iv.data for iv in self.x_tree[x_min:x_max])
        y_hits = set(iv.data for iv in self.y_tree[y_min:y_max])
        candidates = x_hits & y_hits
        return {c for c in candidates if c is not None}

    def rects_overlap(self, pos1, w1, h1, pos2, w2, h2):
        x1_min = pos1[0] - w1 / 2.0
        x1_max = pos1[0] + w1 / 2.0
        y1_min = pos1[1] - h1 / 2.0
        y1_max = pos1[1] + h1 / 2.0

        x2_min = pos2[0] - w2 / 2.0
        x2_max = pos2[0] + w2 / 2.0
        y2_min = pos2[1] - h2 / 2.0
        y2_max = pos2[1] + h2 / 2.0

        return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

    def spawn_rectangle(self, height, width, max_attempts=2000):
        if width >= self.space_width or height >= self.space_height:
            raise ValueError("Rectangle too big for the world")

        min_x = width / 2.0
        max_x = self.space_width - width / 2.0
        min_y = height / 2.0
        max_y = self.space_height - height / 2.0

        for _ in range(max_attempts):
            pos_x = self.rng.random() * (max_x - min_x) + min_x
            pos_y = self.rng.random() * (max_y - min_y) + min_y
            position = np.array([pos_x, pos_y], dtype=float)
            candidates = self.potential_overlaps_for_rect(position, width, height)
            overlap = any(self.rects_overlap(position, width, height, c.position, c.width, c.height) for c in candidates)
            if not overlap:
                return Rectangle(self, self.space, position=position, height=height, width=width)

        raise Exception("Failed to spawn rectangle without overlap after many attempts.")