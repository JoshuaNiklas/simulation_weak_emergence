import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space import PropertyLayer
from mesa_approach.agents.agents import RectangleAgent


class RectangleOptimizationModel(mesa.Model):

    def __init__(
        self, width=100, height=100, seed=None, population_size=0
    ):
        super().__init__(seed=seed)
        self.grid = OrthogonalMooreGrid((width, height), torus=False, random=self.random)
        property_layer = PropertyLayer("occupied", (width, height), default_value=0, dtype=int)
        self.grid.add_property_layer(property_layer)

        for i in range(population_size):
            x = self.random.randint(0, width - 1)
            y = self.random.randint(0, height - 1)
            agent = RectangleAgent(self, id=i, width=2, height=3, pos=(x, y))
            self.agents.add(agent)
            for cell in agent.cells:
                self.grid.occupied.data[cell.coordinate] = agent.id
        
    def step(self):
        """
        Run one step of the model.

        All agents are activated in random order using the AgentSet shuffle_do method.
        """
        self.agents.shuffle_do("step")
