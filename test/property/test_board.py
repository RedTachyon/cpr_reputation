from cpr_reputation.board import HarvestGame

from hypothesis import given
from hypothesis.strategies import integers


@given(
    num_agents=integers(min_value=1, max_value=10),
    size_i=integers(min_value=11, max_value=1000),
    size_j=integers(min_value=11, max_value=1000),
    sight_width=integers(min_value=3, max_value=50),
    sight_dist=integers(min_value=2, max_value=100)
)
def test_get_agent_obs_board_shape(
        num_agents,
        size_i,
        size_j,
        sight_width,
        sight_dist
):
    env1 = HarvestGame(
        num_agents=num_agents,
        size=(size_i, size_j),
        sight_width=sight_width,
        sight_dist=sight_dist
    )
    ag0 = "Agent0"
    constant_obs_shape = (sight_dist, 2 * sight_width + 1, 3)
    env1.agents[ag0].rot = 0 # facing north
    assert env1.get_agent_obs(ag0).shape == constant_obs_shape
    env1.agents[ag0].rot = 1 # facing east
    assert env1.get_agent_obs(ag0).shape == constant_obs_shape
    env1.agents[ag0].rot = 2 # facing south
    assert env1.get_agent_obs(ag0).shape == constant_obs_shape
    env1.agents[ag0].rot = 3 # facing west
    assert env1.get_agent_obs(ag0).shape == constant_obs_shape
