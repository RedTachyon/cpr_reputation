#!/usr/bin/env python3

from copy import deepcopy
import numpy as np
from cpr_reputation.board import (
    apple_values,
    HarvestGame,
    create_board,
    fast_rot90,
    get_neighbors,
    NOOP,
    SHOOT,
    GO_FORWARD,
    GO_BACKWARD,
    GO_LEFT,
    GO_RIGHT,
    #    ROT_LEFT,
    #    ROT_RIGHT,
    DIRECTIONS,
    Position,
    #    HarvestEnv,
)
import pytest

from cpr_reputation.environments import HarvestEnv


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
def example_env1():
    """
    Creates an env with 4 agents in a 4x6 grid.
    1 = apple; 2 = agent

      0 1 2 3 4 5
      -----------
    0|2 2 0 0 1 1
    1|2 2 0 1 0 1
    2|0 0 1 1 1 0
    3|0 0 0 1 0 0

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
def example_env2():
    """10 agents in a 50x50 board, everyone facing south"""
    env = HarvestGame(num_agents=10, size=(50, 50))
    for _, agent in env.agents.items():
        agent.rot = 2
    return env


def test_fast_rot90():
    # should/could be a property-based test
    arr = np.arange(15).reshape((3, 5))
    for k in range(4):
        assert np.all(np.rot90(arr, k) == fast_rot90(arr, k))


def test_get_neighbors_middle():
    neighbors = get_neighbors(Position(2, 2), size=Position(5, 5))
    expected = [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]
    for pos in expected:
        assert pos in neighbors


def test_get_neighbors_corners():
    neighbors = get_neighbors(Position(0, 0), Position(7, 7))
    expected = [(0, 0), (1, 0), (0, 1)]
    for pos in expected:
        assert pos in neighbors

    neighbors = get_neighbors(Position(6, 6), Position(7, 7))
    expected = [(5, 6), (6, 5), (6, 6)]
    for pos in expected:
        assert pos in neighbors


def test_get_neighbors_radius_2():
    neighbors = get_neighbors(Position(4, 3), size=Position(5, 5), radius=2)
    expected = [(2, 3), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]
    for pos in expected:
        assert pos in neighbors


def test_get_agent_obs_board_shape():
    env = HarvestGame(
        size=Position(100, 100), num_agents=1, sight_dist=20, sight_width=10
    )
    env.agents["Agent0"].rot = 0  # facing north
    assert env.get_agent_obs("Agent0").shape == (20, 21, 4)
    env.agents["Agent0"].rot = 1  # facing east
    assert env.get_agent_obs("Agent0").shape == (20, 21, 4)
    env.agents["Agent0"].rot = 2  # facing south
    assert env.get_agent_obs("Agent0").shape == (20, 21, 4)
    env.agents["Agent0"].rot = 3  # facing west
    assert env.get_agent_obs("Agent0").shape == (20, 21, 4)


def test_get_agent_obs_board_items(example_env1):
    env = example_env1
    obs = env.get_agent_obs("Agent3")  # facing south
    obs_apples = np.where(obs[:, :, 0])
    obs_agents = np.where(obs[:, :, 1])
    obs_walls = np.where(obs[:, :, 2])

    expected_apples = [(19, 6), (19, 8), (18, 9), (18, 8), (18, 7), (17, 8)]
    expected_agents = [(t[0], t[1]) for t in [(19, 10), (19, 11)]]
    expected_walls = [
        (t[0] + 1, t[1] + 1)
        for t in [
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (18, 6),
            (18, 11),
            # (19, 6),
            # (19, 11),
        ]
    ]

    # assert sorted(zip(*obs_apples)) == sorted(expected_apples)
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
        (18, 11),
    ]
    expected_agents = [(t[0], t[1]) for t in [(19, 9), (19, 10)]]
    expected_walls = [
        (x - 1, y + 1)
        for x, y in [
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (16, 9),
            (16, 12),
            (17, 9),
            (17, 12),
            (18, 9),
            (18, 12),
            # (19, 9),
            # (19, 12),
        ]
    ]

    # assert sorted(zip(*obs_apples)) == sorted(expected_apples)
    assert sorted(zip(*obs_agents)) == sorted(expected_agents)
    # assert sorted(zip(*obs_walls)) == sorted(expected_walls)

    env._rotate_agent("Agent3", -1)  # facing north
    obs = env.get_agent_obs("Agent3")
    obs_apples = np.where(obs[:, :, 0])
    obs_agents = np.where(obs[:, :, 1])
    obs_walls = np.where(obs[:, :, 2])

    expected_apples = [(18, 13), (18, 14), (19, 12), (19, 14)]
    expected_agents = [(18, 9), (18, 10), (19, 9), (19, 10)]
    expected_walls = [
        (18, 9),
        (18, 10),
        (18, 11),
        (18, 12),
        (18, 13),
        (18, 14),
        (19, 9),
        (19, 14),
    ]

    # assert sorted(zip(*obs_apples)) == sorted(expected_apples)
    # assert sorted(zip(*obs_agents)) == sorted(expected_agents)
    # assert sorted(zip(*obs_walls)) == sorted(expected_walls)

    env._rotate_agent("Agent3", -1)  # facing west
    obs = env.get_agent_obs("Agent3")
    obs_apples = np.where(obs[:, :, 0])
    obs_agents = np.where(obs[:, :, 1])
    obs_walls = np.where(obs[:, :, 2])

    expected_apples = [(19, 10)]
    expected_agents = [(t[0], t[1]) for t in [(18, 10), (18, 11), (19, 10), (19, 11)]]
    expected_walls = [(18, 8), (18, 9), (18, 10), (18, 11), (19, 8), (19, 11)]

    assert sorted(zip(*obs_apples)) == sorted(expected_apples)
    assert sorted(zip(*obs_agents)) == sorted(expected_agents)
    # assert sorted(zip(*obs_walls)) == sorted(expected_walls)


def test_noop(example_env1):
    env = example_env1
    old_agents = deepcopy(env.agents)
    for i in range(4):
        env.process_action(f"Agent{i}", NOOP)
        # Nothing should change
    assert old_agents == env.agents


def test_go_forward(example_env2):
    env = example_env2
    # TODO: make this real
    positions = [agent.pos for _, agent in env.agents.items()]
    for i in range(9, -1, -1):
        agent_id = f"Agent{i}"
        env.process_action(agent_id, GO_FORWARD)
    for position, (name, agent) in zip(positions, env.agents.items()):
        assert position + DIRECTIONS[2] == agent.pos  # south


def test_go_left(example_env1):
    env = example_env1
    positions = [agent.pos for _, agent in env.agents.items()]
    for i in (1, 3, 2, 0):
        env.process_action(f"Agent{i}", GO_LEFT)
    for position, (_, agent) in zip(positions, env.agents.items()):
        assert position + DIRECTIONS[1] == agent.pos


def test_go_right(example_env2):
    env = example_env2
    ag9 = "Agent9"
    env.process_action(ag9, GO_FORWARD)
    env.process_action(ag9, GO_FORWARD)
    old_pos = env.agents[ag9].pos
    env.process_action(ag9, GO_RIGHT)
    assert old_pos + DIRECTIONS[3] == env.agents[ag9].pos


def test_go_forwardbackward_inverses(example_env2):
    env = example_env2
    ag9 = "Agent9"
    old_pos = deepcopy(env.agents[ag9].pos)
    env.process_action(ag9, GO_FORWARD)
    env.process_action(ag9, GO_FORWARD)
    env.process_action(ag9, GO_BACKWARD)
    env.process_action(ag9, GO_BACKWARD)
    assert old_pos == env.agents[ag9].pos


def test_zap_before_50(example_env2):
    env = example_env2
    env.process_action("Agent0", SHOOT)
    # assert env.reputation["Agent0"] == 1
    for i in range(1, 10):
        # assert env.reputation[f"Agent{i}"] == 0
        assert env.agents[f"Agent{i}"].frozen > 0


def test_zap_after_50(example_env2):
    env = example_env2
    env.time = 100
    env.process_action("Agent0", SHOOT)
    # assert env.reputation["Agent0"] == 1
    for i in range(1, 10):
        # assert env.reputation[f"Agent{i}"] == 0
        assert env.agents[f"Agent{i}"].frozen > 0


def test_get_beam_bounds(example_env2):
    env = example_env2
    bound1, bound2 = env.get_beam_bounds("Agent9")
    assert bound1 == (12 + 1, 6 + 1)
    assert bound2 == (2 + 1, -4 + 1)


def test_agent_initial_position():
    env = HarvestGame(num_agents=12, size=Position(5, 5))
    for i in range(env.num_agents):
        assert env.agents[f"Agent{i}"].pos == Position(i // 4 + 1, i % 4 + 1)


def test_apples_do_not_disappear_on_step():
    np.random.seed(1)  # this test is sensitive to randomness
    env = HarvestEnv(config={}, num_agents=1, size=(20, 20))
    env.reset()
    board = deepcopy(env.game.board)
    for step in range(100):
        env.step(actions={"Agent0": NOOP})
    new_board = env.game.board
    assert np.equal(new_board, board).all()  # no new apples should have spawned
    assert new_board.shape == board.shape
    assert new_board.min() == 0
    assert new_board.max() == 1


def test_regenerate_apples_with_step():
    np.random.seed(0)
    env = HarvestEnv(config={}, num_agents=18, size=(20, 20))
    env.reset()

    # start all agents along the west wall, facing east
    for i, agent in enumerate(env.game.agents.values()):
        agent.rot = 1
        agent.pos = Position(i + 1, 1)
    board_beginning = deepcopy(env.game.board)

    # walk towards the east wall
    for i in range(17):
        env.step({agent_id: GO_FORWARD for agent_id in env.game.agents.keys()})
    board_middle = deepcopy(env.game.board)
    assert board_middle.sum() < board_beginning.sum()

    # let some apples regrow
    for step in range(1000):
        env.step({agent_id: NOOP for agent_id in env.game.agents.keys()})
    board_end = deepcopy(env.game.board)
    assert board_end.sum() > board_middle.sum()


def test_apple_values():
    np.random.seed(0)
    env1 = HarvestGame(num_agents=2, size=Position(10, 10))
    assert apple_values(env1.board, Position(7, 3), factor=1) == 8.0
    assert apple_values(env1.board, Position(1, 0), factor=1) == 11.0
