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
        self.velocity = np.zeros(2, dtype=float)

    def step(self):
        COHESION_WEIGHT = 0.4
        SEPARATION_WEIGHT = 1.0
        SHRINK_WEIGHT = 0.6
        ALIGN_WEIGHT = 0.2
        DAMPING = 0.95
        MIN_SPEED = 0.01


        agents = [a for a in self.model.agents if a is not self]

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

            to_center = bbox_center - self.position
            if np.linalg.norm(to_center) > 0:
                cohesion_vec = (to_center / np.linalg.norm(to_center)) * COHESION_WEIGHT
            else:
                cohesion_vec = np.zeros(2)

            bbox_half_diag = np.linalg.norm([
                (bbox_rights.max() - bbox_lefts.min()) / 2.0,
                (bbox_tops.max() - bbox_bottoms.min()) / 2.0
            ])
            dist_to_center = np.linalg.norm(to_center)
            shrink_strength = SHRINK_WEIGHT * (dist_to_center / (bbox_half_diag + 1e-8))
            shrink_vec = (to_center / (dist_to_center + 1e-8)) * shrink_strength \
                if dist_to_center > 0 else np.zeros(2)

            separation_vec = np.zeros(2)
            for other in agents:
                offset = self.position - other.position
                dist = np.linalg.norm(offset)

                min_dx = (self.width + other.width) / 2.0
                min_dy = (self.height + other.height) / 2.0
                min_dist = np.sqrt(min_dx**2 + min_dy**2)

                if dist > 0 and dist < (min_dist * 1.5):
                    separation_vec += (offset / dist) * (SEPARATION_WEIGHT * (1.0 / (dist + 1e-8)))

            align_vec = np.zeros(2)
            neighbor_vels = [a.velocity for a in agents]
            if neighbor_vels:
                avg_vel = np.mean(neighbor_vels, axis=0)
                if np.linalg.norm(avg_vel) > 0:
                    align_vec = (avg_vel / np.linalg.norm(avg_vel)) * ALIGN_WEIGHT

            delta_v = cohesion_vec + shrink_vec + separation_vec + align_vec

        else:
            delta_v = np.zeros(2)

        self.velocity = (self.velocity + delta_v) * DAMPING
        vel_norm = np.linalg.norm(self.velocity)

        if vel_norm > self.speed:
            self.velocity = (self.velocity / vel_norm) * self.speed

        if np.linalg.norm(self.velocity) < MIN_SPEED:
            self.velocity = np.zeros(2)

        new_pos = self.position + self.velocity
        x_min_allowed = self.width / 2.0
        x_max_allowed = self.model.space.x_max - self.width / 2.0
        y_min_allowed = self.height / 2.0
        y_max_allowed = self.model.space.y_max - self.height / 2.0

        new_pos[0] = float(np.clip(new_pos[0], x_min_allowed, x_max_allowed))
        new_pos[1] = float(np.clip(new_pos[1], y_min_allowed, y_max_allowed))

        self.position = new_pos

        try:
            self.model.space.place_agent(self, self.position)
        except Exception:
            pass

        self.resolve_overlaps(agents)

    def resolve_overlaps(self, agents):

        for other in agents:
            left1   = self.position[0] - self.width  / 2
            right1  = self.position[0] + self.width  / 2
            bottom1 = self.position[1] - self.height / 2
            top1    = self.position[1] + self.height / 2

            left2   = other.position[0] - other.width  / 2
            right2  = other.position[0] + other.width  / 2
            bottom2 = other.position[1] - other.height / 2
            top2    = other.position[1] + other.height / 2

            if not (right1 > left2 and left1 < right2 and top1 > bottom2 and bottom1 < top2):
                continue

            overlap_x1 = right2 - left1 
            overlap_x2 = right1 - left2
            overlap_y1 = top2 - bottom1
            overlap_y2 = top1 - bottom2

            overlaps = {
                abs(overlap_x1): np.array([ overlap_x1, 0]),
                abs(overlap_x2): np.array([-overlap_x2, 0]),
                abs(overlap_y1): np.array([0,  overlap_y1]),
                abs(overlap_y2): np.array([0, -overlap_y2]),
            }

            push = overlaps[min(overlaps.keys())]

            self.position += push

            if np.linalg.norm(push) > 0:
                self.velocity -= 0.2 * push

            try:
                self.model.space.place_agent(self, self.position)
            except:
                pass
    

    def create_marker(self):
        """Creates a custom marker for the respective agent"""
        marker = MarkerStyle("s") #Square shape marker
        marker._transform = (marker.get_transform().scale(self.width, self.height))
        return marker
    
    def check_overlap(self, rect2):
        """Check if two rectangles overlap."""
        x1_min = self.position[0] - self.width / 2
        x1_max = self.position[0] + self.width / 2
        y1_min = self.position[1] - self.height / 2
        y1_max = self.position[1] + self.height / 2

        x2_min = rect2.position[0] - rect2.width / 2
        x2_max = rect2.position[0] + rect2.width / 2
        y2_min = rect2.position[1] - rect2.height / 2
        y2_max = rect2.position[1] + rect2.height / 2

        return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)
