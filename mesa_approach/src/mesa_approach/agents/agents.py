"""A rectangle agent for solving the Optimal Rectangle Packing Model


"""

import numpy as np

from mesa.experimental.continuous_space import ContinuousSpaceAgent


class Rectangle(ContinuousSpaceAgent):
    """A rectangle agent.

    The agent follows the following behavior:
        - Cohesion: is drawn closer to other rectangles nearby
        - No-Overlap: overlapping is not permitted

    Rectangles have a direction they move towards to which is represented by a numpy-array
    """

    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        speed=1,
        direction=(1, 1),
        separation=1,
        cohere=0.03,
        radius=1
    ):
        """Create a new Rectangle agent.

        Args:
            model: Model instance the agent belongs to
            speed: Distance to move per step
            direction: numpy vector for the Boid's direction of movement
            vision: Radius to look around for nearby Boids
            separation: Minimum distance to maintain from other Boids
            cohere: Relative importance of matching neighbors' positions (default: 0.03)
            separate: Relative importance of avoiding close neighbors (default: 0.015)
            match: Relative importance of matching neighbors' directions (default: 0.05)
        """
        super().__init__(space, model)
        self.position = position
        self.speed = speed
        self.direction = direction
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.neighbors = []
        self.angle = 0.0  # represents the angle at which the boid is moving

    def step(self):
        """Get the Rectangles's neighbors, compute the new vector, and move accordingly."""
        neighbors, distances = self.get_neighbors_in_radius(radius=self.radius)
        self.neighbors = [n for n in neighbors if n is not self]

        # If no neighbors, maintain current direction
        if not neighbors:
            # Do nothing? self.position += self.direction * self.speed
            return

        delta = self.space.calculate_difference_vector(self.position, agents=neighbors)

        cohere_vector = delta.sum(axis=0) * self.cohere_factor
        separation_vector = (
            -1 * delta[distances < self.separation].sum(axis=0) * self.separate_factor
        )
        match_vector = (
            np.asarray([n.direction for n in neighbors]).sum(axis=0) * self.match_factor
        )

        # Update direction based on the three behaviors
        self.direction += (cohere_vector + separation_vector + match_vector) / len(neighbors)

        # Normalize direction vector (ein Vektor ist dann normalisiert, wenn man ihn durch seine Länge (d.h. seine Norm) teilt!)
        self.direction = self.direction / np.linalg.norm(self.direction)

        # Move rectangles
        self.position += self.direction * self.speed
