#!/usr/bin/env python3

from copy import deepcopy
import numpy as np
from board import (
    HarvestGame,
    create_board, fast_rot90, get_neighbors, regenerate_apples, random_board,
    NOOP, SHOOT, GO_FORWARD, GO_BACKWARD, GO_LEFT, GO_RIGHT, ROT_LEFT, ROT_RIGHT,
    DIRECTIONS,
    Position, HarvestEnv
    )
import matplotlib.pyplot as plt
import pytest


def test_init():
    env = HarvestGame(num_agents=10, size=(50, 50))
    assert env.board.shape == (50, 50)
    for i in range(10):
        assert f"Agent{i}" in env.agents


def test_create():
    apples = [(5, 5), (0, 0), (10, 20)]
    board = create_board((50, 50), apples)
    assert board.shape == (50, 50)

    all_apples = [get_neighbors(Position(*apple), Position(50, 50)) for apple in apples]
    all_apples = [pos for apple in all_apples for pos in apple]

    for i in range(50):
        for j in range(50):
            assert board[i, j] == ((i, j) in all_apples)

@pytest.fixture
def example_env1_nowalls():
    """
    Creates an env with 4 agents in a 4x6 grid.
    1 = apple; 2 = agent

      0 1 2 3 4 5
      -----------
    0|2 2 0 0 1 1
    1|2 2 0 1 0 1
    2|0 0 1 1 1 0
    3|0 0 0 1 0 0
    4|0 0 0 0 0 0


    - All agents start facing south
    - No walls
    """

    size = (4, 6)
    apples = [(0, 5), (2, 3)]
    env = HarvestGame(num_agents=4, size=size)

    env.board = create_board(size, apples)
    for agent_id, agent in env.agents.items():
        agent.rot = 2  # south
    return env

@pytest.fixture
def example_env2_nowalls():
    """10 agents in a 50x50 board, everyone facing south"""
    env = HarvestGame(num_agents=10, size=(50, 50))
    for _, agent in env.agents.items():
        agent.rot = 2
    return env

def test_fast_rot90():
    # should/could be a property-based test
    arr = np.arange(15).reshape((3, 5))
    for k in range(4):
        assert(np.all(np.rot90(arr, k) == fast_rot90(arr, k)))


def test_get_neighbors_middle():
    neighbors = get_neighbors(Position(2, 2), size=Position(5, 5))
    expected = [
        (2, 2),
        (1, 2),
        (3, 2),
        (2, 1),
        (2, 3)
    ]
    for pos in expected:
        assert pos in neighbors


def test_get_neighbors_corners():
    neighbors = get_neighbors(Position(0, 0), Position(7, 7))
    expected = [
        (0, 0),
        (1, 0),
        (0, 1)
    ]
    for pos in expected:
        assert pos in neighbors

    neighbors = get_neighbors(Position(6, 6), Position(7, 7))
    expected = [
        (5, 6),
        (6, 5),
        (6, 6)
    ]
    for pos in expected:
        assert pos in neighbors


def test_get_neighbors_radius_2():
    neighbors = get_neighbors(Position(4, 3), size=Position(5, 5), radius=2)
    expected = [
        (2, 3),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        (4, 3),
        (4, 4)
    ]
    for pos in expected:
        assert pos in neighbors


def test_regrow():  # TODO make deterministic
    np.random.seed(0)
    board = random_board((50, 50), prob=0.2)
    new_board = regenerate_apples(board)  # This will fail once we fix the regrowing
    assert np.sum(new_board) > np.sum(board)
    assert new_board.shape == board.shape
    assert new_board.min() == 0
    assert new_board.max() == 1

    # No apples should disappear
    for i in range(50):
        for j in range(50):
            if board[i, j] > 0:
                assert new_board[i, j] > 0


def test_get_agent_obs_board_shape():
    env = HarvestGame(size=Position(100, 100), num_agents=1)
    env.agents["Agent0"].rot = 0  # facing north
    assert env.get_agent_obs("Agent0").shape == (20, 21, 3)
    env.agents["Agent0"].rot = 1  # facing east
    assert env.get_agent_obs("Agent0").shape == (20, 21, 3)
    env.agents["Agent0"].rot = 2  # facing south
    assert env.get_agent_obs("Agent0").shape == (20, 21, 3)
    env.agents["Agent0"].rot = 3  # facing west
    assert env.get_agent_obs("Agent0").shape == (20, 21, 3)


def test_get_agent_obs_board_items(example_env1_nowalls):  # TODO add walls
    env = example_env1_nowalls
    obs = env.get_agent_obs("Agent3")  # facing south
    obs_apples = np.where(obs[:, :, 0])
    obs_agents = np.where(obs[:, :, 1])
    obs_walls = np.where(obs[:, :, 2])

    expected_apples = [
        (19, 6),
        (19, 8),
        (18, 9),
        (18, 8),
        (18, 7),
        (17, 8)
    ]
    expected_agents = [
        (19, 10),
        (19, 11)
    ]
    expected_walls = []

    assert sorted(zip(*obs_apples)) == sorted(expected_apples)
    assert sorted(zip(*obs_agents)) == sorted(expected_agents)
    assert sorted(zip(*obs_walls)) == sorted(expected_walls)

    env._rotate_agent("Agent3", -1)  # facing east
    obs = env.get_agent_obs("Agent3")
    obs_apples = np.where(obs[:, :, 0])
    obs_agents = np.where(obs[:, :, 1])
    obs_walls = np.where(obs[:, :, 2])

    expected_apples = [
        (15, 9),
        (15, 10),
        (16, 9),
        (16, 11),
        (17, 10),
        (17, 11),
        (17, 12),
        (18, 11)
    ]
    expected_agents = [
        (19, 9),
        (19, 10)
    ]
    expected_walls = []

    assert sorted(zip(*obs_apples)) == sorted(expected_apples)
    assert sorted(zip(*obs_agents)) == sorted(expected_agents)
    assert sorted(zip(*obs_walls)) == sorted(expected_walls)

    env._rotate_agent("Agent3", -1)  # facing north
    obs = env.get_agent_obs("Agent3")
    obs_apples = np.where(obs[:, :, 0])
    obs_agents = np.where(obs[:, :, 1])
    obs_walls = np.where(obs[:, :, 2])

    expected_apples = [
        (18, 13),
        (18, 14),
        (19, 12),
        (19, 14)
    ]
    expected_agents = [
        (18, 9),
        (18, 10),
        (19, 9),
        (19, 10)
    ]
    expected_walls = []

    assert sorted(zip(*obs_apples)) == sorted(expected_apples)
    assert sorted(zip(*obs_agents)) == sorted(expected_agents)
    assert sorted(zip(*obs_walls)) == sorted(expected_walls)

    env._rotate_agent("Agent3", -1)  # facing west
    obs = env.get_agent_obs("Agent3")
    obs_apples = np.where(obs[:, :, 0])
    obs_agents = np.where(obs[:, :, 1])
    obs_walls = np.where(obs[:, :, 2])

    expected_apples = []
    expected_agents = [
        (18, 10),
        (18, 11),
        (19, 10),
        (19, 11)
    ]
    expected_walls = []

    assert sorted(zip(*obs_apples)) == sorted(expected_apples)
    assert sorted(zip(*obs_agents)) == sorted(expected_agents)
    assert sorted(zip(*obs_walls)) == sorted(expected_walls)


def test_step(example_env2_nowalls):
    env = example_env2_nowalls
    old_agents = deepcopy(env.agents)
    for i in range(10):
        env.process_action(f"Agent{i}", NOOP)
        # Nothing should change
    assert old_agents == env.agents

    env.process_action("Agent0", SHOOT)
    assert env.reputation["Agent0"] == 1
    for i in range(1, 10):
        assert env.reputation[f"Agent{i}"] == 0

    positions = [agent.pos for _, agent in env.agents.items()]
    for i in range(10):
        agent_id = f"Agent{i}"
        agent = env.agents[agent_id]
        env.process_action(agent_id, GO_FORWARD)
    for position, (name, agent) in zip(positions, env.agents.items()):
        assert position + DIRECTIONS[2] == agent.pos # down

    # TODO: More step-wise tests


def test_zap(example_env2_nowalls):
    env = example_env2_nowalls

    pass  # TODO: duh

def test_get_beam_bounds():
    pass # TODO

def test_process_action():
    pass # TODO

def test_move_agent():
    # TODO
    pass

def test_agent_initial_position():
    # TODO
    pass
