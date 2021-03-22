#!/usr/bin/env python3

from copy import deepcopy
import numpy as np
from board import HarvestGame, create_board, fast_rot90, get_neighbors, regenerate_apples, random_board, NOOP, SHOOT, Position
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


def test_fast_rot90():
    arr = np.arange(15).reshape((3, 5))
    for k in range(4):
        assert(np.rot90(arr, k) == fast_rot90(arr, k))


def test_neighbors():
    neighbors = get_neighbors(Position(2, 2), Position(5, 5))
    expected = [
        (2, 2),
        (1, 2),
        (3, 2),
        (2, 1),
        (2, 3)
    ]
    assert len(neighbors) == 5
    for pos in expected:
        assert pos in neighbors

    neighbors = get_neighbors(Position(0, 0), Position(7, 7))
    expected = [
        (0, 0),
        (1, 0),
        (0, 1)
    ]
    assert len(neighbors) == 3
    for pos in expected:
        assert pos in neighbors


def test_regrow():
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


def test_step():
    env = HarvestGame(num_agents=10, size=(50, 50))
    old_agents = deepcopy(env.agents)
    for i in range(10):
        env.process_action(f"Agent{i}", NOOP)
        # Nothing should change
    assert old_agents == env.agents

    env.process_action("Agent0", SHOOT)
    assert env.reputation["Agent0"] == 1
    for i in range(1, 10):
        assert env.reputation[f"Agent{i}"] == 0

    # TODO: More step-wise tests


def test_obs():
    env = HarvestGame(num_agents=1, size=(100, 100), sight_dist=20, sight_width=10)
    # Agent is in the middle, should have full vision in each direction
    env._move_agent("Agent0", Position(50, 50))
    for direction in range(4):
        env._set_rotation("Agent0", direction)
        assert env.get_agent_obs("Agent0").shape == (20, 21, 3)


def test_zap():
    pass  # TODO: duh
