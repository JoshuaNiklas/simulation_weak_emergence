import numpy as np
from constants import SIMULATION_SIZE

class Agent:

    def __init__(self, id, position_init, color, shared_memory):
        self.id = id
        self.position = position_init
        self.color = color
        self.shared_memory = shared_memory

    def move(self):
        others = self.read_memory()
        self_center = np.mean(self.position, axis=0)

        move_vector = np.array([0.0, 0.0])
        overlap_found = False

        # Check for overlap with others
        for other_pos in others:
            if self.is_overlapping(other_pos):
                # Move away from center of overlapping rectangle
                self_center = np.mean(self.position, axis=0)
                other_center = np.mean(other_pos, axis=0)
                direction = self_center - other_center
                norm = np.linalg.norm(direction)
                if norm != 0:
                    direction /= norm
                move_vector += direction
                overlap_found = True

            if not overlap_found:
                density = self.local_density(others, radius=150)
                if density > 3:
                    avg_pos = np.mean([np.mean(p, axis=0) for p in others], axis=0)
                    direction = self_center - avg_pos
                    if np.linalg.norm(direction) != 0:
                        direction /= np.linalg.norm(direction)
                    move_vector += direction
                else:
                    direction = -self_center
                    if np.linalg.norm(direction) != 0:
                        direction /= np.linalg.norm(direction)
                    move_vector += 0.2 * direction

            step_size = 2.0
            self.position[0] = list(np.array(self.position[0]) + step_size * move_vector)
            self.position[1] = list(np.array(self.position[1]) + step_size * move_vector)
        self.position = self.clamp_position()
        self.write_memory()

    def is_overlapping(self, other_pos):
        x1_min, y1_min = self.position[0]
        x1_max, y1_max = self.position[1]
        x2_min, y2_min = other_pos[0]
        x2_max, y2_max = other_pos[1]

        return not (x1_max < x2_min or x1_min > x2_max or
                    y1_max < y2_min or y1_min > y2_max)

    def clamp_position(self):
        width = self.position[1][0] - self.position[0][0]
        height = self.position[1][1] - self.position[0][1]

        # Stelle sicher, dass linke untere Ecke im gültigen Bereich bleibt
        x0 = np.clip(self.position[0][0], 0, SIMULATION_SIZE - width)
        y0 = np.clip(self.position[0][1], 0, SIMULATION_SIZE - height)

        # Obere rechte Ecke ergibt sich aus Größe
        x1 = x0 + width
        y1 = y0 + height

        return [[x0, y0], [x1, y1]]

    def read_memory(self):
        memory = []
        for agent_id in self.shared_memory:
            if(agent_id != self.id):
                entry = self.shared_memory[agent_id]
                memory.append(entry)
        return memory

    def write_memory(self):
        self.shared_memory[self.id] = self.position

    def local_density(self, others, radius=150):
        center = np.mean(self.position, axis=0)
        count = 0
        for pos in others:
            other_center = np.mean(pos, axis=0)
            if np.linalg.norm(center - other_center) < radius:
                count += 1
        return count


    
            