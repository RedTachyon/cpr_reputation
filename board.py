from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
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
                  radius=1) -> List[Position]:
    """Get a list of neighboring positions (including self) that are still within the boundaries of the board"""
    row0, col0 = pos
    return [(row1, col1) for col1 in range(size[1]) for row1 in range(size[0]) if abs(row1 - row0) + abs(col1 - col0) <= radius]


def create_board(size: Position,
                 initial_apples: List[Position]) -> Board:
    """Creates an empty board with a number of cross-shaped apple patterns"""
    orig_board = np.zeros(size)
    for (row, col) in initial_apples:
        orig_board[row][col] = 1

    board = orig_board.copy()

    for row in range(len(orig_board)):
        for col in range(len(orig_board[0])):
            if orig_board[row][col]:
                for (row_neighbor, col_neighbor) in get_neighbors((row, col), size, radius=1):
                    board[row_neighbor, col_neighbor] = 1
    return board


def random_board(size: Position,
                 num_crosses=10) -> Board:
    """Creates a board with random cross-shaped apple patterns"""
    all_positions = [(row, col) for col in range(len(size[1])) for row in range(len(size[0]))]
    initial_apples = np.random.choice(all_positions, num_crosses)
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


class HarvestGame:
    def __init__(self,
                 num_agents: int,
                 size: Position,
                 sight_width: Optional[int] = 10,  # default value from DM paper
                 sight_dist: Optional[int] = 20  # default value from DM paper
                 ):

        self.num_agents = num_agents
        self.size = size
        self.sight_width = sight_width
        self.sight_dist = sight_dist

        self.board = np.array([[]])
        self.agents = {}
        self.reputation = {}

        self.reset()

    def reset(self):
        self.board = random_board(self.size, num_crosses=10)
        self.agents = {
            f"Agent{i}": Walker(pos=agent_initial_position(i, self.num_agents), rot=1+np.random.randint(2))  # start facing east or south
            for i in range(self.num_agents)
        }

        # remove apples from cells which already contain agents
        for name, agent in self.agents.items():
            self.board[agent.pos] = 0

        self.reputation = {
            f"Agent{i}": 0
            for i in range(self.num_agents)
        }

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

    """
    def regenerate_apples(self):
        Performs a single update of stochastically regenerating apples.
        It works by computing a convolution counting neighboring apples, scaling this with a base probability,
        and using that as the probability of an apple being generated there.
        Needs to be adapted to the version in the DM paper. Also might need an agent-based mask.
        kernel = NEIGHBOR_KERNEL
        neighbor_map: np.ndarray = convolve(self.board, kernel, mode='constant')
        prob_map = neighbor_map * self.regen_prob

        rand = np.random.rand(*neighbor_map.shape)
        regen_map = rand < prob_map
        updated_board = np.clip(self.board + regen_map, 0, 1)

        # don't spawn apples in cells which already contain agents
        for name, agent in self.agents.items():
            updated_board[agent.pos] = 0

        self.board = updated_board
    """

    def regenerate_apples(self):
        """
        Stochastically respawn apples based on number of neighbors:

        prob | n_neighbors
        ------------------
        0    | L = 0
        0.01 | L = 1 or 2
        0.05 | L = 3 or 4
        0.1  | L > 4

        Could probably be faster...
        """
        prob_table = defaultdict(lambda: 0.1)
        prob_table[0] = 0.0
        prob_table[1] = 0.01
        prob_table[2] = 0.01
        prob_table[3] = 0.05
        prob_table[4] = 0.05

        prob_map = np.zeros(self.size)
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                pos = (row, col)
                neighboring_pos = get_neighbors(pos, self.size, radius=2)
                neighboring_apples = sum([self.board[r][c] for (r, c) in neighboring_pos])
                prob_map[row][col] = prob_table[neighboring_apples]

        rand = np.random.rand(prob_map.shape)
        regen_map = rand < prob_map
        updated_board = np.clip(self.board + regen_map, 0, 1)

        # don't spawn apples in cells which already contain agents
        for name, agent in self.agents.items():
            updated_board[agent.pos] = 0

        self.board = updated_board

    def process_action(self, agent_id: str, action: int) -> float:
        """Processes a single action for a single agent"""
        agent = self.agents[agent_id]
        (row, col), rot = agent.pos, agent.rot
        if agent.frozen > 0:  # if the agent is frozen, should we avoid updating the gradient?
            agent.frozen -= 1
            return 0.
        if action == GO_FORWARD:
            # Go forward
            (drow, dcol) = DIRECTIONS[rot]
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
            new_pos = (row, col)
        elif action == ROT_RIGHT:
            # Rotate right
            self._rotate_agent(agent, 1)
            new_pos = (row, col)
        elif action == SHOOT:
            # Shoot a beam
            affected_agents = self.get_affected_agents(agent_id)
            for _agent in affected_agents:
                _agent.frozen = 25
            self.reputation[agent_id] += 1  # whrow does reputation increase after shooting?
            new_pos = (row, col)
            # see notebook for weird results
        elif action == 7:
            # No-op
            new_pos = (row, col)

        row_new, col_new = new_pos[0], new_pos[1]
        if self.board[row_new][col_new]:  # apple in new cell
            self.board[row_new][col_new] = 0
            return 1
        else:  # no apple in new cell
            return 0

    def get_affected_agents(self, agent_id: str) -> List[Walker]:
        """Returns a list of agents caught in the ray"""
        (row0, col0), (row1, col1) = self.get_observable_window(agent_id)
        return [other_agent for other_name, other_agent in self.agents.items()
                if other_name != agent_id
                and row0 <= other_agent.pos[0] <= row1
                and col0 <= other_agent.pos[1] <= col1]

    def get_observable_window(self, agent_id: str) -> List[Position]:
        """Returns indices of the (top left, bottom right) (inclusive) boundaries of an agent's vision."""
        row_max = self.size[0] - 1
        col_max = self.size[1] - 1
        agent = self.agents[agent_id]
        row, col= agent.pos
        if agent.rot == 0:  # facing north
            bound_left = max(0, col - self.sight_width)
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
        return [(bound_left, bound_up), (bound_right, bound_down)]

    def get_agent_obs(self, agent_id: str) -> np.ndarray:
        """The partial observability of the environment.
        TODO: this should have been TDD, so the TODO is to write some simple test cases and make sure they pass.
        """
        (row0, col0), (row1, col1) = self.get_observable_window(agent_id)
        return self.board[row0:row1+1][col0:col1+1]


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
