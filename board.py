from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
from numba import vectorize
from scipy.ndimage import convolve

import matplotlib as mpl
import matplotlib.pyplot as plt

GO_FORWARD = 0
GO_BACKWARD = 1
GO_LEFT = 2
GO_RIGHT = 3
ROT_LEFT = 4
ROT_RIGHT = 5
SHOOT = 6
NOOP = 7

# Kernel to compute the number of surrounding apples - not the same as in DM paper
NEIGHBOR_KERNEL = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
])

DIRECTIONS = [
    (-1, 0),  # UP
    (0, 1),  # RIGHT
    (1, 0),  # DOWN
    (0, -1),  # LEFT
]

# noinspection PyUnresolvedReferences
cmap = mpl.colors.ListedColormap(['white', 'red', 'green'])

Board = np.ndarray
Position = Tuple[int, int]


@dataclass
class Walker:
    pos: Position
    rot: int
    frozen: int = 0
    #
    # def __repr__(self) -> str:
    #     return f"Agent(pos={self.pos}, frozen={self.frozen > 0})"


def in_bounds(pos: Position,
              size: Position) -> bool:
    """Checks whether the position fits within the board of a given size"""
    (row, col) = pos
    (height, width) = size
    return 0 <= row < height and 0 <= col < width


def get_neighbors(pos: Position,
                  size: Position,
                  radius: int = 1) -> List[Position]:
    """Get a list of neighboring positions (including self) that are still within the boundaries of the board"""
    if radius == 1:  # Default
        increments = [
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        (x, y) = pos

        return [(x + dx, y + dy) for (dx, dy) in increments if in_bounds((x + dx, y + dy), size)]
    else:
        row0, col0 = pos
        return [(row1, col1) for col1 in range(size[1]) for row1 in range(size[0]) if
                abs(row1 - row0) + abs(col1 - col0) <= radius]


def create_board(size: Position,
                 apples: List[Position]) -> Board:
    """Creates an empty board with a number of cross-shaped apple patterns"""
    board = np.zeros(size, dtype=np.int8)

    for pos in apples:
        for (x, y) in get_neighbors(pos, size, radius=1):
            board[x, y] = 1

    return board


def random_board(size: Position,
                 prob: float = 0.1) -> Board:
    """Creates a board with each square having a probability of containing an apple"""
    prob_map = np.random.rand(*size)
    return (prob_map < prob).astype(np.int8)


def random_crosses(size: Position,
                   num_crosses: int = 10) -> Board:
    """Creates a board with random cross-shaped apple patterns"""
    all_positions = [(row, col) for col in range(size[1]) for row in range(size[0])]
    random_idx = np.random.choice(range(len(all_positions)), num_crosses, replace=False)
    initial_apples = [all_positions[i] for i in random_idx]
    board = create_board(size, initial_apples)
    return board


def agent_initial_position(i: int, total: int) -> Position:
    """
    Finds a starting position for an i-th agent with $total agents overall.
    The idea is that they're aligned in a square in the (0, 0) corner.
    The size of the square is determined by the number of agents.
    For example, with 10 agents, they will be in a 4x4 square, with 6 spots unfilled.
    With 16 agents, they're in a full 4x4 square.
    """
    layout_base = int(np.ceil(np.sqrt(total)))
    idx_map = np.arange(layout_base ** 2).reshape(layout_base, layout_base)
    (rows, cols) = np.where(idx_map == i)
    row, col = rows[0], cols[0]
    return row, col


@vectorize
def regrow_prob(n: int) -> float:
    if n == 0:
        return 0.0
    elif n == 1 or n == 2:
        return 0.01
    elif n == 3 or n == 4:
        return 0.05
    else:
        return 0.1


def regenerate_apples(board: Board) -> Board:
    kernel = NEIGHBOR_KERNEL
    neighbor_map: np.ndarray = convolve(board, kernel, mode='constant')  # TODO: make sure this is ints
    prob_map = regrow_prob(neighbor_map)

    rand = np.random.rand(*neighbor_map.shape)
    regen_map = rand < prob_map
    updated_board = np.clip(board + regen_map, 0, 1)

    return updated_board


class HarvestGame:
    def __init__(self,
                 num_agents: int,
                 size: Position,
                 sight_width: Optional[int] = 10,  # default value from DM paper
                 sight_dist: Optional[int] = 20,  # default value from DM paper
                 num_crosses: Optional[int] = 10  # number of apple-crosses to start with
                 ):

        self.num_agents = num_agents
        self.size = size
        self.sight_width = sight_width
        self.sight_dist = sight_dist
        self.num_crosses = num_crosses

        self.board = np.array([[]])
        self.agents = {}
        self.reputation = {}

        self.reset()

    def reset(self):
        self.board = random_crosses(self.size, self.num_crosses)
        self.agents = {
            f"Agent{i}": Walker(pos=agent_initial_position(i, self.num_agents), rot=1 + np.random.randint(2))
            # start facing east or south
            for i in range(self.num_agents)
        }

        # remove apples from cells which already contain agents
        for name, agent in self.agents.items():
            self.board[agent.pos] = 0

        self.reputation = {
            f"Agent{i}": 0
            for i in range(self.num_agents)
        }

    def render(self, ax: plt.Axes):  # TODO: make it usable without an axes
        """Writes the image to a pyplot axes"""
        board = self.board[:]
        for name, agent in self.agents.items():
            board[agent.pos] = 2
        ax.cla()
        ax.imshow(board, cmap=cmap)

    def is_free(self, pos: Position) -> bool:  # TODO: Check if it's on a wall
        """Checks whether the position is within bounds, and unoccupied"""
        if not in_bounds(pos, self.size):
            return False

        for agent_id, agent in self.agents.items():
            if agent.pos == pos:
                return False

        return True

    def _move_agent(self, agent_id: str, new_pos: Position):
        """Maybe moves an agent to a new position"""
        agent = self.agents[agent_id]
        if self.is_free(new_pos):
            agent.pos = new_pos

    def _set_rotation(self, agent_id: str, rot: int):
        """Debugging/testing method"""
        agent = self.agents[agent_id]
        assert rot in (0, 1, 2, 3)
        agent.rot = rot

    def _rotate_agent(self, agent_id: str, direction: int):
        """Rotates an agent left or right"""
        agent = self.agents[agent_id]
        assert direction in [-1, 1]
        # -1: rotate left
        #  1: rotate right
        agent.rot = (agent.rot + direction) % 4

    def process_action(self, agent_id: str, action: int) -> float:
        """Processes a single action for a single agent"""
        agent = self.agents[agent_id]
        (row, col), rot = agent.pos, agent.rot
        if agent.frozen > 0:  # if the agent is frozen, should we avoid updating the gradient?
            agent.frozen -= 1
            return 0.
        if action == GO_FORWARD:
            # Go forward
            (drow, dcol) = DIRECTIONS[rot]  # TODO Maybe change to (i, j)
            new_pos = (row + drow, col + dcol)
            self._move_agent(agent, new_pos)
        elif action == GO_BACKWARD:
            # Go backward
            (drow, dcol) = DIRECTIONS[rot]
            new_pos = (row - drow, col - dcol)
            self._move_agent(agent, new_pos)
        elif action == GO_LEFT:
            # Go left
            (drow, dcol) = DIRECTIONS[(rot - 1) % 4]
            new_pos = (row + drow, col + dcol)
            self._move_agent(agent, new_pos)
        elif action == GO_RIGHT:
            # Go right
            (drow, dcol) = DIRECTIONS[(rot + 1) % 4]
            new_pos = (row + drow, col + dcol)
            self._move_agent(agent, new_pos)
        elif action == ROT_LEFT:
            # Rotate left
            self._rotate_agent(agent, -1)
        elif action == ROT_RIGHT:
            # Rotate right
            self._rotate_agent(agent, 1)
        elif action == SHOOT:
            # Shoot a beam
            affected_agents = self.get_affected_agents(agent_id)
            for _agent in affected_agents:
                _agent.frozen = 25
            self.reputation[agent_id] += 1  # whrow does reputation increase after shooting?
            # see notebook for weird results
        elif action == NOOP:
            # No-op
            pass
        else:
            raise ValueError(f"Invalid action {action}")

        current_pos = self.agents[agent_id].pos
        if self.board[current_pos]:  # apple in new cell
            self.board[current_pos] = 0
            return 1.
        else:  # no apple in new cell
            return 0.

    def get_affected_agents(self, agent_id: str) -> List[Walker]: # FIXME: change the behavior, set a parameter for beam width/length
        """Returns a list of agents caught in the ray"""
        (row0, col0), (row1, col1) = self.get_beam_bounds(agent_id)
        return [other_agent for other_name, other_agent in self.agents.items()
                if other_name != agent_id
                and row0 <= other_agent.pos[0] <= row1
                and col0 <= other_agent.pos[1] <= col1]

    def get_agent_obs(self, agent_id: str):
        agent = self.agents[agent_id]
        apple_board = self.board
        agent_board = np.zeros_like(self.board)
        for other_agent_id, other_agent in self.agents.items():
            agent_board[other_agent.pos] = 1

        wall_board = ...

        all_boards = np.stack([apple_board, agent_board, wall_board], axis=-1)  # W x H x 3
        rot = self.agents[agent_id].rot

        board = np.rot90(all_boards, rot)  # TODO: make sure it's in the right direction

        agent_i, agent_j = agent.pos
        bounds_i = (agent_i - 19, agent_i + 1)  # TODO: maybe +1
        bounds_j = (agent_j - 10, agent_j + 11)  # TODO: OB1 errors
        # board[i:j] == board[slice(i, j)]
        base_slice = board[slice(*bounds_i), slice(*bounds_j)]  # (20, 21) OR (10, 21)
        if bounds_i[0] < 0:  # TODO: Complete this
            base_slice = np.concatenate([base_slice, np.zeros(...)])
        if bounds_j[0] < 0:
            base_slice = np.concatenate([base_slice, np.zeros((base_slice.shape[0], -bounds_j[0], base_slice.shape[2]))], axis=1)
        if bounds_j[1] > self.board.shape[1]:
            base_slice = np.concatenate([base_slice, np.zeros(...)], axis=1)
        if bounds_i[1] > self.board.shape[0]:
            raise ValueError("WTF")

        return base_slice  # 20 x 21

    def get_beam_bounds(self, agent_id: str) -> List[Position]:
        """Returns indices of the (top left, bottom right) (inclusive) boundaries of an agent's vision."""
        row_max = self.size[0] - 1
        col_max = self.size[1] - 1
        agent = self.agents[agent_id]
        row, col = agent.pos

        if agent.rot == 0:  # facing north
            bound_left = col - self.sight_width
            bound_right = min(col_max, col + self.sight_width)
            bound_up = max(0, row - self.sight_dist)
            bound_down = row
        elif agent.rot == 1:  # facing east
            bound_left = col
            bound_right = min(col_max, col + self.sight_dist)
            bound_up = max(0, row - self.sight_width)
            bound_down = min(row_max, row + self.sight_width)
        elif agent.rot == 2:  # facing south
            bound_left = max(0, col - self.sight_width)
            bound_right = min(col_max, col + self.sight_width)
            bound_up = row
            bound_down = min(row_max, row + self.sight_dist)
        elif agent.rot == 3:  # facing west
            bound_left = max(0, col - self.sight_dist)
            bound_right = col
            bound_up = max(0, row - self.sight_width)
            bound_down = min(row_max, row + self.sight_width)
        else:
            raise ValueError("agent.rot must be % 4")
        return [(bound_up, bound_left), (bound_down, bound_right)]  # (top left, bottom right)



class MultiAgentEnv:  # Placeholder
    pass


class HarvestEnv(MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        self.time = 0
        self.game = HarvestGame(*args, **kwargs)

    def reset(self) -> Dict[str, np.ndarray]:
        self.game.reset()
        self.time = 0
        return {
            agent_id: self.game.get_agent_obs(agent_id) for agent_id in self.game.agents
        }

    def step(self, actions: Dict[str, int]):
        rewards = {agent_id: 0. for agent_id in self.game.agents}
        for (agent_id, action) in actions.items():
            reward = self.game.process_action(agent_id, action)
            rewards[agent_id] += reward

        obs = {
            agent_id: self.game.get_agent_obs(agent_id) for agent_id in self.game.agents
        }

        done = {
            agent_id: self.time > 1000 for agent_id in self.game.agents
        }
        done["__all__"] = self.time > 1000  # Required for rllib (I think)
        info = {}
        self.time += 1

        return obs, rewards, done, info
