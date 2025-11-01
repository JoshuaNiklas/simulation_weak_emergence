import numpy as np

class Agent:

    def __init__(self, id, position_init, color, shared_memory):
        self.id = id
        self.position = position_init
        self.color = color
        self.shared_memory = shared_memory

    def move(self):
        pass

    def read_memory(self):
        memory = []
        for agent_id in self.shared_memory:
            if(agent_id != self.id):
                entry = self.shared_memory[agent_id]
                memory.append(entry)
        return memory

    def write_memory(self):
        self.shared_memory[self.id] = self.position

    
            