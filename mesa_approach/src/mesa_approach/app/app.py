from mesa_approach.model.model import RectangleOptimizationModel
from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle


def agent_portrayal(agent):

    return AgentPortrayalStyle(
        color="red", marker="s", size=agent.width * agent.height
    )


# Model parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
}

# Initialize model
initial_model = RectangleOptimizationModel(width=10, height=10, population_size=1)
# Create grid and agent visualization component using Altair
renderer = SpaceRenderer(initial_model, backend="matplotlib").render(agent_portrayal)

# Create visualization with all components
page = SolaraViz(
    model=initial_model,
    renderer=renderer,
    model_params=model_params,
    name="Rectangle Space Optiomization Model",
)
page  # noqa B018
