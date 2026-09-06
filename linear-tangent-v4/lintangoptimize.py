import logging
from typing import Callable

import numpy as np
from scipy.linalg import norm

import constants
import integrator
from data_types import State, Vector3D, WorldConfig, ShipConfig
from dynamics_examples import get_state_orbital

l = logging.getLogger("l")


def simulate_dynamics_lintan(
    altitude: np.float64,
    velocity: np.float64,
    time: np.float64,
    dt: float,
    c_a: np.float64,
    c_b: np.float64,
    c_t: np.float64,
    log: int = 0,
) -> list[tuple[float, State]]:
    initial_state, world_cfg, ship_cfg = get_state_orbital(altitude, velocity)
    initial_state.extra_data = {"thrust": [0, 0, 0], "angle": 0}

    i_f = int((np.ceil(time / dt)) + 1)
    result = np.ndarray((i_f, 2), dtype=object)

    cur_state = initial_state
    t = 0
    i = 0
    result[0] = np.array([0, initial_state], dtype=object)

    init_ground = initial_state.pos - world_cfg.moon_center
    init_vel = initial_state.vel

    init_ground_norm = init_ground / np.linalg.norm(init_ground)
    init_vel_perp = init_vel - np.dot(init_vel, init_ground_norm) * init_ground_norm
    init_vel_perp_norm = init_vel_perp / np.linalg.norm(init_vel_perp)

    terminal_pitch = False
    reference_cur_vel = None

    while t < time:
        if cur_state.altitude(world_cfg) < 100:
            break
        if cur_state.extra_mass <= 0:
            break

        theta = np.rad2deg(np.arctan(c_a * (t - c_t) + c_b))
        m = ship_cfg.engine_thrust * (t >= c_t)

        cur_ground = cur_state.pos - world_cfg.moon_center
        cur_vel = cur_state.vel
        reference_cur_vel = cur_vel if not terminal_pitch else reference_cur_vel

        if norm(cur_vel) < 200:
            terminal_pitch = True

        cur_ground_proj = (
            np.dot(cur_ground, init_ground_norm) * init_ground_norm
            + np.dot(cur_ground, init_vel_perp_norm) * init_vel_perp_norm
        )
        c_hat = cur_ground_proj / np.linalg.norm(cur_ground_proj)

        alpha = np.deg2rad(90.0 + theta)
        cos_phi = (
            np.linalg.norm(cur_ground) / np.linalg.norm(cur_ground_proj)
        ) * np.cos(alpha)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.arccos(cos_phi)

        c2d = np.array(
            [np.dot(c_hat, init_ground_norm), np.dot(c_hat, init_vel_perp_norm)]
        )
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        u2d_plus = R @ c2d
        u2d_minus = R.T @ c2d

        u_plus = u2d_plus[0] * init_ground_norm + u2d_plus[1] * init_vel_perp_norm
        u_minus = u2d_minus[0] * init_ground_norm + u2d_minus[1] * init_vel_perp_norm

        cur_vel_proj = (
            np.dot(reference_cur_vel, init_ground_norm) * init_ground_norm
            + np.dot(reference_cur_vel, init_vel_perp_norm) * init_vel_perp_norm
        )
        if np.dot(u_plus, cur_vel_proj) >= np.dot(u_minus, cur_vel_proj):
            u = u_plus
        else:
            u = u_minus

        thrust: Vector3D = -m * u

        cur_state = integrator.integrate_state(
            cur_state, world_cfg, ship_cfg, thrust, dt
        )
        t += dt
        i += 1
        cur_state.extra_data = {
            "thrust": thrust.tolist(),
            "angle": float(theta),
        }
        result[i] = np.array([t, cur_state], dtype=object)

        if log == 2 and i % 100 == 0 or log == 1 and i % round(i_f / 100) == 0:
            l.info(
                "{}% {}s Simulation - Alt {}m; Vel {}m/s ; Fuel {}kg ; Angle {}deg".format(
                    round(i / i_f * 100, 2),
                    round(t, 3),
                    round(cur_state.altitude(world_cfg), 0),
                    round(norm(cur_state.vel), 1),
                    round(cur_state.extra_mass, 2),
                    round(theta, 2),
                )
            )

    result[i + 1 :] = np.array([-1, None], dtype=object)

    return result.tolist()


def lintang_orbit_test(
    dt: np.float64,
    a: np.float64,
    b: np.float64,
    t: np.float64,
    log: int = 2,
) -> tuple[list[tuple[float, State]], WorldConfig, ShipConfig]:
    altitude = constants.INITIAL_ALTITUDE
    velocity = constants.INITIAL_SPEED
    time_f = 1000

    l.info("Starting simulation")
    l.info(
        " - Altitude {}km ; Velocity {}m/s ; Simulating {} minutes at dt {}s with a b t {} {} {}".format(
            round(altitude / 1000, 3),
            round(velocity),
            round(time_f / 60, 1),
            dt,
            round(a, 2),
            round(b, 2),
            round(t, 2),
        )
    )
    data = simulate_dynamics_lintan(altitude, velocity, time_f, dt, a, b, t, log)
    initial_state, world_cfg, ship_cfg = get_state_orbital(altitude, velocity)

    data = list(filter(lambda x: x[0] >= 0, data))
    last_state: State = data[-1][1]

    l.info("Finished simulation")
    l.info(
        " - Altitude {}km ; Velocity {}m/s ; Simulating {} minutes at dt {}s with a b t {} {} {}".format(
            round(last_state.altitude(world_cfg) / 1000, 3),
            round(norm(last_state.vel), 1),
            round(data[-1][0] / 60, 1),
            dt,
            round(a, 2),
            round(b, 2),
            round(t, 2),
        )
    )

    return data, world_cfg, ship_cfg
