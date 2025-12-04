from mesa import Agent
from mesa.discrete_space.cell_collection import CellCollection
from mesa.discrete_space import CellAgent

class RectangleAgent(Agent):

    def __init__(self, model, id, width=2, height=3, pos=(0, 0)):
        super().__init__(model)
        self.id = id
        self.width = width
        self.height = height
        self.pos = pos  # bottom-left corner
        self.cells = self.create_cell_collection(width, height, pos)


    def create_cell_collection(self, width, height, bottom_left_pos):
        """Create a CellCollection representing the rectangle."""
        cell_coordinates = []
        x0, y0 = bottom_left_pos
        for dx in range(width):
            for dy in range(height):
                cell_pos = (x0 + dx, y0 + dy)
                cell_coordinates.append(cell_pos)
        return self.model.grid.all_cells.select(filter_func=lambda cell: cell.coordinate in cell_coordinates, at_most=width*height)
        

    def step(self):
        self.move((1, 0))  # Example: move right by 1 unit

    def move(self, direction):
        self.cells = self.create_cell_collection(self.width, self.height, self.pos + direction)

