from simulation import Simulation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import argparse
from constants import *

def simulate(timestep):
    simulation.run(timestep)
    for rectangle, agent in zip(rectangles.values(), simulation.agents):
        rectangle.set_xy(agent.position[0])
        rectangle.set_width(agent.position[1][0] - agent.position[0][0])
        rectangle.set_height(agent.position[1][1] - agent.position[0][1])
    return rectangles.values()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Q&D-Simulation of Multi-Agent-System")
    arg_parser.add_argument("--size", type=int, default=SIMULATION_SIZE, help="Size of the Simulation")
    arg_parser.add_argument("--speed", type=float, default=DT, help="Speed of the Simulation")
    arg_parser.add_argument("--number", type=int, default=N_AGENTS, help="Number of Agents within Simulation")
    args = arg_parser.parse_args()
    simulation_size = args.size
    simulation_speed = args.speed
    n_agents = args.number

    simulation = Simulation(simulation_size, simulation_speed, n_agents)
    fig, ax = plt.subplots()
    ax.set_xlim(0, simulation_size)
    ax.set_ylim(0, simulation_size)

    #Set up animation
    rectangles = {agent.id: patches.Rectangle(
        agent.position[0], agent.position[1][0] - agent.position[0][0], agent.position[1][1] - agent.position[0][1],
        facecolor=agent.color, alpha=0.5) for agent in simulation.agents}

    for id in rectangles:
        ax.add_patch(rectangles[id])


    animation = FuncAnimation(fig, simulate, frames=200, interval=20, blit=False)
    plt.show()