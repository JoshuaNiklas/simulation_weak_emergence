from matplotlib.markers import MarkerStyle
from mesa_approach.model.model import OptimalRectanglePackingModel
from mesa.visualization import Slider, SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle




def rectangle_draw(agent):
    rectangle_style = AgentPortrayalStyle(
        color="red", size=20, marker=agent.marker
    )

    return rectangle_style


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "population_size": Slider(
        label="Number of rectangles",
        value=5,
        min=2,
        max=10,
        step=1,
    ),
    "width": 100,
    "height": 100,
    "speed": Slider(
        label="Speed of rectangle movement",
        value=5,
        min=1,
        max=20,
        step=1,
    ),
}

model = OptimalRectanglePackingModel()

# Quickest way to visualize grid along with agents or property layers.
renderer = SpaceRenderer(
    model,
    backend="matplotlib",
).render(agent_portrayal=rectangle_draw)

page = SolaraViz(
    model,
    renderer,
    model_params=model_params,
    name="Optimal Rectangle Packing Model",
)
page  # noqa
