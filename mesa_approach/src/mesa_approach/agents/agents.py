"""A rectangle agent for solving the Optimal Rectangle Packing Model
"""

import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent
from matplotlib.markers import MarkerStyle


class Rectangle(ContinuousSpaceAgent):
    """A rectangle agent.

    The agent follows the following behavior:
        - Cohesion: is drawn closer to other rectangles nearby
        - No-Overlap: overlapping is not permitted via sliding (no bounce)

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
        """Create a new Rectangle agent."""
        super().__init__(space, model)
        self.position = np.array(position, dtype=float)
        self.speed = float(speed)
        self.height = float(height)
        self.width = float(width)
        self.marker = self.create_marker()
        self.velocity = np.zeros(2, dtype=float)

    def create_marker(self):
        """Creates a custom marker for the respective agent"""
        marker = MarkerStyle("s")  # Square shape marker
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

    def step(self):
        """Move with cohesion + shrink + separation, forbidding overlaps by sliding (no bounce)."""
        COHESION_WEIGHT = 0.4
        SEPARATION_WEIGHT = 1.0
        SHRINK_WEIGHT = 0.6
        ALIGN_WEIGHT = 0.2
        DAMPING = 0.95
        MIN_SPEED = 0.01

        all_agents = [a for a in self.model.agent_list if a is not self]

        if len(all_agents) > 0:
            positions = np.array([a.position for a in self.model.agent_list])
            bbox_lefts = np.array([a.position[0] - a.width / 2.0 for a in self.model.agent_list])
            bbox_rights = np.array([a.position[0] + a.width / 2.0 for a in self.model.agent_list])
            bbox_bottoms = np.array([a.position[1] - a.height / 2.0 for a in self.model.agent_list])
            bbox_tops = np.array([a.position[1] + a.height / 2.0 for a in self.model.agent_list])

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
            candidates = self.model.potential_overlaps_for_rect(self.position, self.width, self.height)
            for other in candidates:
                if other is self:
                    continue
                offset = self.position - other.position
                dist = np.linalg.norm(offset)

                min_dx = (self.width + other.width) / 2.0
                min_dy = (self.height + other.height) / 2.0
                min_dist = np.sqrt(min_dx**2 + min_dy**2)

                if dist > 0 and dist < (min_dist * 1.5):
                    separation_vec += (offset / dist) * (SEPARATION_WEIGHT * (1.0 / (dist + 1e-8)))

            align_vec = np.zeros(2)
            neighbor_vels = [a.velocity for a in all_agents]
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

        # Collision-free movement: try full move, then axis-only, then binary search
        old_pos = self.position.copy()
        pos = old_pos
        vel = self.velocity

        if np.allclose(vel, 0.0):
            return

        def overlaps_at(position):
            """Check overlap at position using interval trees."""
            candidates = self.model.potential_overlaps_for_rect(position, self.width, self.height)
            candidates = [c for c in candidates if c is not self]
            return any(self.model.rects_overlap(position, self.width, self.height, c.position, c.width, c.height) for c in candidates)

        full_pos = pos + vel
        if not overlaps_at(full_pos):
            new_pos = full_pos
        else:
            x_only = pos + np.array([vel[0], 0.0])
            y_only = pos + np.array([0.0, vel[1]])
            blocked_x = overlaps_at(x_only)
            blocked_y = overlaps_at(y_only)

            if not blocked_x:
                new_pos = x_only
                self.velocity[1] = 0.0
            elif not blocked_y:
                new_pos = y_only
                self.velocity[0] = 0.0
            else:
                low, high = 0.0, 1.0
                for _ in range(12):
                    mid = 0.5 * (low + high)
                    if overlaps_at(pos + vel * mid):
                        high = mid
                    else:
                        low = mid
                t_allowed = low
                if t_allowed <= 1e-8:
                    new_pos = pos
                    self.velocity[:] = 0.0
                else:
                    new_pos = pos + vel * t_allowed
                    small = 1e-6
                    if overlaps_at(pos + np.array([vel[0] * (t_allowed + small), 0.0])):
                        self.velocity[0] = 0.0
                    if overlaps_at(pos + np.array([0.0, vel[1] * (t_allowed + small)])):
                        self.velocity[1] = 0.0

        x_min_allowed = self.width / 2.0
        x_max_allowed = self.model.space.x_max - self.width / 2.0
        y_min_allowed = self.height / 2.0
        y_max_allowed = self.model.space.y_max - self.height / 2.0
        new_pos[0] = float(np.clip(new_pos[0], x_min_allowed, x_max_allowed))
        new_pos[1] = float(np.clip(new_pos[1], y_min_allowed, y_max_allowed))

        self.position = new_pos
        # Update interval trees
        try:
            self.model.update_agent_intervals(self, old_position=old_pos)
        except Exception:
            pass