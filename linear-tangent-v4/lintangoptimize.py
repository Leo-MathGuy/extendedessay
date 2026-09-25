import logging

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
    result = np.ndarray((i_f + 1, 2), dtype=object)

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
    last_alt = altitude

    while t < time:
        if cur_state.altitude(world_cfg) < 100:
            break

        theta = np.rad2deg(np.arctan(c_a * (t - c_t) + c_b))
        m = ship_cfg.engine_thrust * (t >= c_t)

        cur_ground = cur_state.pos - world_cfg.moon_center
        cur_vel = cur_state.vel
        reference_cur_vel = cur_vel if not terminal_pitch else reference_cur_vel

        cur_ground_norm = cur_ground / np.linalg.norm(cur_ground)
        horizontal_vel = cur_vel - np.dot(cur_vel, cur_ground_norm) * cur_ground_norm
        if norm(horizontal_vel) < 200:
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

        u = u / (norm(u) + 1e-15)
        engine = ship_cfg.engine_thrust

        def thrust_fn(t_abs):
            return -engine * u

        cur_state = integrator.integrate_state_rk4_thrusted(
            cur_state, world_cfg, ship_cfg, thrust_fn, t, dt, c_t
        )

        if cur_state.altitude(world_cfg) > last_alt + 0.1:
            break

        t += dt
        i += 1
        cur_state.extra_data = {
            "thrust": thrust_fn(t).tolist(),
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

        last_alt = cur_state.altitude(world_cfg)

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
    time_f = 1800

    l.info("Starting simulation")
    l.info(
        " - Altitude {}km ; Velocity {}m/s ; Simulating {} minutes at dt {}s with a b t {} {} {}".format(
            round(altitude / 1000, 3),
            round(velocity),
            round(time_f / 60, 1),
            dt,
            round(a, 4),
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
            round(a, 4),
            round(b, 2),
            round(t, 2),
        )
    )

    return data, world_cfg, ship_cfg


def get_cost_penalty(final_state: State, world: WorldConfig) -> np.float64:
    alt = final_state.altitude(world)
    angle = 90 - final_state.extra_data["angle"]

    # l.info(
    #     f"Alt {alt:.1f} -> {(alt - 100) ** 2:.1f}; Vel {norm(final_state.vel):.1f} -> {2 * ((norm(final_state.vel)) ** 2):.1f}; Angle {angle:.1f} -> {20 * (angle**3):.1f}"
    # )

    return ((alt) ** 2) + 2 * ((norm(final_state.vel)) ** 2) + 20 * (angle**2)


def find_derivative(
    altitude: np.float64,
    velocity: np.float64,
    time_f: np.float64,
    a: np.float64,
    b: np.float64,
    t: np.float64,
    t_diff: np.float64,
    dt: np.float64,
    log: int,
) -> tuple[np.float64, np.float64, np.float64]:
    # Left

    data_l = simulate_dynamics_lintan(
        altitude, velocity, time_f, dt, a, b, t - t_diff, log
    )
    initial_state, world_cfg, ship_cfg = get_state_orbital(altitude, velocity)

    data_l = list(filter(lambda x: x[0] >= 0, data_l))
    last_state_l: State = data_l[-1][1]
    tf_l = data_l[-1][0]
    f_l = get_cost_penalty(last_state_l, world_cfg)

    angle_l = 90 - last_state_l.extra_data["angle"]

    # Right

    data_r = simulate_dynamics_lintan(
        altitude, velocity, time_f, dt, a, b, t + t_diff, log
    )
    initial_state, world_cfg, ship_cfg = get_state_orbital(altitude, velocity)

    data_r = list(filter(lambda x: x[0] >= 0, data_r))
    last_state_r: State = data_r[-1][1]
    tf_r = data_r[-1][0]
    f_r = get_cost_penalty(last_state_r, world_cfg)

    angle_r = 90 - last_state_r.extra_data["angle"]

    # Derivative
    der = (f_r - f_l) / (t_diff * 2)

    return der, f_l, f_r


def do_a_b_pair(
    dt: np.float64, a: np.float64, b: np.float64, init_time: np.float64, log: int = 2
) -> tuple[list[tuple[float, State]], WorldConfig, ShipConfig]:
    altitude = constants.INITIAL_ALTITUDE
    velocity = constants.INITIAL_SPEED
    time_f = 1800

    iteration = 0
    max_iterations = 15

    t_diff = 1

    t = init_time

    if log:
        l.info("Starting simulation")
        l.info(
            " - Altitude {}km ; Velocity {}m/s ; Simulating {} minutes at dt {}s with a b {} {}".format(
                round(altitude / 1000, 3),
                round(velocity),
                round(time_f / 60, 1),
                dt,
                round(a, 4),
                round(b, 2),
            )
        )

    test_der = np.arange(0, np.ceil(init_time / 100) * 100 + 1, 100)

    derivatives_hundereds = np.ndarray((len(test_der), 2), dtype=float)

    for i, t in enumerate(test_der):
        d, _, _ = find_derivative(altitude, velocity, time_f, a, b, t, t_diff, dt, log)
        derivatives_hundereds[i] = np.array([t, d])

    if min(derivatives_hundereds[:, 1]) > 0 or max(derivatives_hundereds[:, 1]) < 0:
        l.error("Cannot find a zero interval")
        return None

    sign_changes = np.where(np.diff(np.sign(derivatives_hundereds[:, 1])))[0]

    change = sign_changes[0]

    bound_l = derivatives_hundereds[change, 0]
    bound_r = derivatives_hundereds[change + 1, 0]
    if log:
        l.info(f"Sign changes between t {bound_l} and {bound_r}")

    t = bound_r

    test_min = np.linspace(bound_l, bound_r, max_iterations)
    values = np.ndarray((len(test_min),), dtype=float)

    initial_state, world_cfg, ship_cfg = get_state_orbital(altitude, velocity)

    for i, t in enumerate(test_min):
        data_t = simulate_dynamics_lintan(altitude, velocity, time_f, dt, a, b, t, log)
        data_t = list(filter(lambda x: x[0] >= 0, data_t))
        last_state_t: State = data_t[-1][1]
        tf_t = data_t[-1][0]
        f_t = get_cost_penalty(last_state_t, world_cfg)
        values[i] = f_t
        if log:
            l.info(f"Minimum rough found {t:.2f}={f_t:.1f}")

    target = values.min() * 0.8

    while True:
        if log:
            l.info(f"Performing iteration {iteration} t = {t}")

        data_c = simulate_dynamics_lintan(altitude, velocity, time_f, dt, a, b, t, log)

        data_c = list(filter(lambda x: x[0] >= 0, data_c))

        penalty = get_cost_penalty(data_c[-1][1], world_cfg)
        data_c[-1][1].extra_data["penalty"] = penalty

        last_state_c: State = data_c[-1][1]
        tf_c = data_c[-1][0]
        f_c = get_cost_penalty(last_state_c, world_cfg)

        angle_c = 90 - last_state_c.extra_data["angle"]

        if log:
            l.info(
                f"Ended at {tf_c:.2f}: Angle {round(last_state_c.extra_data["angle"], 2)} - {round(angle_c, 2)} from vertical; f = {f_c:.2f}"
            )

        iteration += 1
        if iteration > max_iterations:
            break

        new_time = -1
        der, f_l, f_r = find_derivative(
            altitude, velocity, time_f, a, b, t, t_diff, dt, log
        )
        for i, current_damping in enumerate([0.6, 0.2, 0.03, 0.005, 0.0005]):
            new_time = t - current_damping * (f_c - target) / der
            if log:
                l.info(
                    f"Adjusting by {current_damping} * {(f_c-target):.2f} / {der:.2f} -> {t:.1f} -> {new_time:.1f}"
                )

            if bound_l > new_time or new_time > bound_r:
                continue

            data_N = simulate_dynamics_lintan(
                altitude, velocity, time_f, dt, a, b, new_time, log
            )
            initial_state, world_cfg, ship_cfg = get_state_orbital(altitude, velocity)

            data_N = list(filter(lambda x: x[0] >= 0, data_N))
            last_state_N: State = data_N[-1][1]
            tf_N = data_N[-1][0]
            f_N = get_cost_penalty(last_state_N, world_cfg)

            if f_N > f_c:
                if log:
                    l.info("Cost increasing, increasing damping")
                continue

            if bound_l < new_time < bound_r:
                break

        if abs(t - new_time) < dt / 2:
            break

        t = new_time

        if t < bound_l:
            l.error(f"New time {t} below")
            return None
        if t > bound_r:
            l.error(f"New time {t} over")
            return None

    if last_state_c.altitude(world_cfg) > 1000:
        l.info("Failed to find one that lands")
        return None

    if log:
        l.info("Finished simulation")
        l.info(
            " - Altitude {}km ; Velocity {}m/s ; Simulated {} minutes at dt {}s with a b t {} {} {}".format(
                round(last_state_c.altitude(world_cfg) / 1000, 3),
                round(norm(last_state_c.vel), 1),
                round(data_c[-1][0] / 60, 1),
                dt,
                round(a, 2),
                round(b, 2),
                round(t, 2),
            )
        )

    return data_c, world_cfg, ship_cfg
