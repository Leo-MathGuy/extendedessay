import logging

import numpy as np
from numpy import linalg

import constants
import integrator
from data_types import WorldConfig, ShipConfig, State

l = logging.getLogger("l")


def get_state_orbital(
    altitude: np.float64, velocity: np.float64
) -> tuple[State, WorldConfig, ShipConfig]:
    state = State(
        np.float64(altitude + constants.MOON_RADIUS),
        0,
        0,
        0,
        np.float64(velocity),
        0,
        np.float64(constants.MIN_MASS),
        np.float64(constants.EXTRA_PROPELLANT_MASS),
    )

    world_config = WorldConfig(
        moon_center=np.array([0, 0, 0], dtype=np.float64),
        moon_mass=np.float64(constants.MOON_MASS),
        moon_radius=np.float64(constants.MOON_RADIUS),
    )
    ship_config = ShipConfig(
        engine_thrust=np.float64(constants.ENGINE_FORCE),
        engine_isp=np.float64(constants.ENGINE_ISP),
    )

    return state, world_config, ship_config


def simulate_dynamics(
    altitude: np.float64,
    velocity: np.float64,
    time: np.float64,
    dt: float,
    log: int = 0,
) -> list[State]:
    initial_state, world_cfg, ship_cfg = get_state_orbital(altitude, velocity)

    i_f = (np.ceil(time / dt)) + 1
    result = np.ndarray((i_f, 2), dtype=object)

    cur_state = initial_state
    t = 0
    i = 0
    result[0] = np.array([0, initial_state], dtype=object)
    while t < time:
        cur_state = integrator.integrate_state(
            cur_state, world_cfg, ship_cfg, np.array([0, 0, 0]), dt
        )
        t += dt
        i += 1
        result[i] = np.arange([t, cur_state], dtype=object)

        if log == 2 and i % 50 or log == 1 and i % round(i / i_f / 100) == 0:
            l.info(
                "{}% {}s Simulation - Alt {} ; Vel {}".format(
                    round(i / i_f * 100, 2),
                    round(t, 3),
                    cur_state.altitude(world_cfg),
                    linalg.norm(cur_state.vel),
                )
            )

    result[i + 1 :] = np.array([-1, None], dtype=object)

    return result
