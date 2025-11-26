"""A rectangle agent for solving the Optimal Rectangle Packing Model


"""

import numpy as np

from mesa.experimental.continuous_space import ContinuousSpaceAgent
from matplotlib.markers import MarkerStyle


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
        height=1,
        width=1
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
        self.height = height
        self.width = width
        self.neighbors = []
        self.marker = self.create_marker()

    def step(self):
        """Move the rectangle towards a compact configuration (cohesion + shrink) while avoiding overlaps."""
        # hyperparameters (tune these)
        COHESION_WEIGHT = 0.4
        SEPARATION_WEIGHT = 2.0
        SHRINK_WEIGHT = 0.6
        ALIGN_WEIGHT = 0.2
        DAMPING = 0.95  # reduces movement to let system settle
        MIN_SPEED = 0.01

        # create velocity attribute if missing
        if not hasattr(self, "velocity"):
            self.velocity = np.zeros(2, dtype=float)

        agents = [a for a in self.model.agents if a is not self]

        # compute center-of-mass of all agents (cohesion)
        if len(agents) > 0:
            positions = np.array([a.position for a in self.model.agents])
            bbox_lefts = np.array([a.position[0] - a.width / 2.0 for a in self.model.agents])
            bbox_rights = np.array([a.position[0] + a.width / 2.0 for a in self.model.agents])
            bbox_bottoms = np.array([a.position[1] - a.height / 2.0 for a in self.model.agents])
            bbox_tops = np.array([a.position[1] + a.height / 2.0 for a in self.model.agents])

            bbox_center = np.array([
                (bbox_lefts.min() + bbox_rights.max()) / 2.0,
                (bbox_bottoms.min() + bbox_tops.max()) / 2.0,
            ])

            # cohesion vector (toward bbox center)
            to_center = bbox_center - self.position
            if np.linalg.norm(to_center) > 0:
                cohesion_vec = (to_center / np.linalg.norm(to_center)) * COHESION_WEIGHT
            else:
                cohesion_vec = np.zeros(2)

            # shrink vector: push toward center but stronger if we are at the periphery
            # distance to bbox center relative to half-diagonal for scaling
            bbox_half_diag = np.linalg.norm([ (bbox_rights.max() - bbox_lefts.min()) / 2.0,
                                             (bbox_tops.max() - bbox_bottoms.min()) / 2.0 ])
            dist_to_center = np.linalg.norm(to_center)
            shrink_strength = SHRINK_WEIGHT * (dist_to_center / (bbox_half_diag + 1e-8))
            shrink_vec = (to_center / (dist_to_center + 1e-8)) * shrink_strength if dist_to_center > 0 else np.zeros(2)

            # separation: avoid overlaps and close neighbours
            separation_vec = np.zeros(2)
            for other in agents:
                # vector away from other; use centers
                offset = self.position - other.position
                dist = np.linalg.norm(offset)
                # minimal allowed distance (sum of half widths/heights as approximation)
                min_dx = (self.width + other.width) / 2.0
                min_dy = (self.height + other.height) / 2.0
                # treat Euclidean minimal distance as average of dx/dy
                min_dist = np.sqrt(min_dx**2 + min_dy**2)
                if dist > 0 and dist < (min_dist * 1.5):  # threshold for repulsion
                    # stronger repulsion when closer
                    separation_vec += (offset / dist) * (SEPARATION_WEIGHT * (1.0 / (dist + 1e-8)))

            # optional alignment: align velocity with neighbors
            align_vec = np.zeros(2)
            neighbor_vels = [getattr(a, "velocity", np.zeros(2)) for a in agents]
            if neighbor_vels:
                avg_vel = np.mean(neighbor_vels, axis=0)
                if np.linalg.norm(avg_vel) > 0:
                    align_vec = (avg_vel / np.linalg.norm(avg_vel)) * ALIGN_WEIGHT

            # combine forces into velocity change
            delta_v = cohesion_vec + shrink_vec + separation_vec + align_vec

        else:
            # no other agents: do nothing
            delta_v = np.zeros(2)

        # update velocity and apply damping
        self.velocity = (self.velocity + delta_v) * DAMPING

        # limit step length to speed
        speed = getattr(self, "speed", 1.0)
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm > speed:
            self.velocity = (self.velocity / vel_norm) * speed

        # if very slow then stop (settle)
        if np.linalg.norm(self.velocity) < MIN_SPEED:
            self.velocity = np.zeros(2)

        # compute tentative new position and clamp so rectangle stays fully inside bounds
        new_pos = self.position + self.velocity
        x_min_allowed = self.width / 2.0
        x_max_allowed = self.model.space.x_max - self.width / 2.0
        y_min_allowed = self.height / 2.0
        y_max_allowed = self.model.space.y_max - self.height / 2.0

        new_pos[0] = float(np.clip(new_pos[0], x_min_allowed, x_max_allowed))
        new_pos[1] = float(np.clip(new_pos[1], y_min_allowed, y_max_allowed))

        # apply position (and update space if you use ContinuousSpace placement tracking)
        self.position = new_pos
        try:
            self.model.space.place_agent(self, self.position)
        except Exception:
            # fallback if place_agent is not available or incompatible
            pass

    def create_marker(self):
        """Creates a custom marker for the respective agent"""
        marker = MarkerStyle("s") #Square shape marker
        marker._transform = (marker.get_transform().scale(self.width, self.height))
        return marker
