from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
from scipy.ndimage import convolve

import matplotlib as mpl
import matplotlib.pyplot as plt

# Kernel to compute the number of surrounding apples - not the same as in DM paper
NEIGHBOR_KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

DIRECTIONS = [
    (-1, 0),  # UP
    (0, 1),   # RIGHT
    (1, 0),   # DOWN
    (0, -1),  # LEFT
]

# noinspection PyUnresolvedReferences
cmap = mpl.colors.ListedColormap(['white', 'red', 'green'])

Board = np.ndarray
Position = Tuple[int, int]


@dataclass
class Agent:
    pos: Position
    rot: int


def in_bounds(pos: Position,
              size: Position) -> bool:
    """Checks whether the position fits within the board of a given size"""
    (x, y) = pos
    (w, h) = size
    return 0 <= x < w and 0 <= y < h


def neighbors(pos: Position,
              size: Position) -> List[Position]:
    """Get a list of neighboring positions (including self) that are still within the boundaries of the board"""
    increments = [
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    (x, y) = pos

    return [(x + dx, y + dy) for (dx, dy) in increments if in_bounds((x + dx, y + dy), size)]


def create_board(size: Position,
                 apples: List[Position]) -> Board:
    """Creates an empty board with a number of cross-shaped apple patterns"""
    board = np.zeros(size, dtype=np.int8)

    for pos in apples:
        for (x, y) in neighbors(pos, size):
            board[x, y] = 1

    return board


def random_board(size: Position,
                 prob: float = 0.1) -> Board:
    """Creates a board with each square having a probability of containing an apple"""
    prob_map = np.random.rand(*size)
    return (prob_map < prob).astype(np.int8)


def regenerate_apples(board: Board, prob: float = 0.1) -> Board:
    """
    Performs a single update of stochastically regenerating apples.
    It works by computing a convolution counting neighboring apples, scaling this with a base probability,
    and using that as the probability of an apple being generated there.
    Needs to be adapted to the version in the DM paper. Also might need an agent-based mask.
    """
    kernel = NEIGHBOR_KERNEL
    neighbor_map: np.ndarray = convolve(board, kernel, mode='constant')
    prob_map = neighbor_map * prob

    rand = np.random.rand(*neighbor_map.shape)

    regen_map = rand < prob_map

    updated_board = np.clip(board + regen_map, 0, 1)
    return updated_board


def initial_position(i: int, total: int) -> Position:
    """
    Finds a starting position for an i-th agent with $total agents overall.
    The idea is that they're aligned in a square in the (0, 0) corner.
    The size of the square is determined by the number of agents.
    For example, with 10 agents, they will be in a 4x4 square, with 6 spots unfilled.
    With 16 agents, they're in a full 4x4 square.
    """
    layout_base = int(np.ceil(np.sqrt(total)))
    idx_map = np.arange(layout_base**2).reshape(layout_base, layout_base)
    (x, y) = np.where(idx_map == i)
    x, y = x[0], y[0]
    return x, y


class HarvestEnv:
    def __init__(self,
                 num_agents: int,
                 size: Position,
                 init_prob: float = 0.1,
                 regen_prob: float = 0.01):

        self.size = size
        self.init_prob = init_prob
        self.regen_prob = regen_prob

        self.board = np.array([[]])

        self.num_agents = num_agents
        self.agents = {}

        self.reset()

    def reset(self):
        self.board = random_board(self.size, self.init_prob)
        self.agents = {
            f"Agent{i}": Agent(pos=initial_position(i, self.num_agents), rot=np.random.randint(4))
            for i in range(self.num_agents)
        }

        for name, agent in self.agents.items():
            self.board[agent.pos] = 0

    def render(self, ax: plt.Axes):
        """Writes the image to a pyplot axes"""
        board = self.board[:]
        for name, agent in self.agents.items():
            board[agent.pos] = 2
        ax.cla()
        ax.imshow(board, cmap=cmap)

    def is_free(self, pos: Position) -> bool:
        """Checks whether the position is within bounds, and unoccupied"""
        if not in_bounds(pos, self.size):
            return False

        for agent in self.agents:
            if agent.pos == pos:
                return False

        return True

    def _move_agent(self, agent: Agent, new_pos: Position):
        """Maybe moves an agent to a new position"""
        if self.is_free(new_pos):
            agent.pos = new_pos

    def _rotate_agent(self, agent: Agent, direction: int):
        """Rotates an agent left or right"""
        assert direction in [-1, 1]
        # -1: rotate left
        #  1: rotate right
        agent.rot = (agent.rot + direction) % 4

    def _process_action(self, agent: Agent, action: int):
        """Processes a single action for a single agent"""
        (x, y), rot = agent.pos, agent.rot
        if action == 0:
            # Go forward
            (dx, dy) = DIRECTIONS[rot]
            new_pos = (x + dx, y + dy)
            self._move_agent(agent, new_pos)
        elif action == 1:
            # Go backward
            (dx, dy) = DIRECTIONS[rot]
            new_pos = (x - dx, y - dy)
            self._move_agent(agent, new_pos)
        elif action == 2:
            # Go left
            (dx, dy) = DIRECTIONS[(rot - 1) % 4]
            new_pos = (x + dx, y + dy)
            self._move_agent(agent, new_pos)
        elif action == 3:
            # Go left
            (dx, dy) = DIRECTIONS[(rot + 1) % 4]
            new_pos = (x + dx, y + dy)
            self._move_agent(agent, new_pos)
        elif action == 4:
            # Rotate left
            self._rotate_agent(agent, -1)
        elif action == 5:
            # Rotate right
            self._rotate_agent(agent, 1)
        elif action == 6:
            # Shoot a beam, to implement
            pass
        elif action == 7:
            # No-op
            pass

    def _get_affected_agents(self, agent: Agent, length: int = 6, width: int = 10) -> List[Agent]:
        """Returns a list of agents caught in the ray"""
        pass


    def _agent_obs(self, agent: Agent) -> np.ndarray:
        pass


