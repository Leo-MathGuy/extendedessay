import random

import numpy as np
from scipy.constants import gravitational_constant
from scipy.linalg import norm

from data_types import State, WorldConfig


def test_state_basic():
    vector = [random.randint(-10000000, 10000000) for x in range(6)]
    vector += [500, 100]

    state = State(*vector)

    assert state.pos.tolist() == [vector[0], vector[1], vector[2]]
    assert state.vel.tolist() == [vector[3], vector[4], vector[5]]

    assert state.mass == 500 + 100


def test_state_derivative():
    vector = [1000, 1000, 1000, 50, 50, 50, 100, 50]
    world_config = WorldConfig(np.array([0, 0, 0]), 1000, 100)

    radius_squared = 3000000
    gravity_accel = 1000 * gravitational_constant / radius_squared

    state = State(*vector)

    v1 = state.dx_dt(world_config, np.array([0, 0, 0]))

    assert v1[:3].tolist() == vector[3:6]
    assert round(norm(v1[3:6])) == round(gravity_accel)


def test_state_derivative_under_thrust():
    vector = [1000, 0, 0, 50, 50, 50, 100, 100]
    world_config = WorldConfig(np.array([0]), 1000, 100)

    radius_squared = 1000000
    gravity_accel = 1000 * gravitational_constant / radius_squared

    state = State(*vector)

    accel_vector = np.array([-1, 0, 0]) * gravity_accel
    thrust = np.array([10, -10, 0])

    v1 = state.dx_dt(world_config, thrust)

    assert v1[:3].tolist() == vector[3:6]

    assert v1[3] == (accel_vector[0] + 10 / 200)
    assert v1[4] == -10 / 200
    assert v1[5] == 0
