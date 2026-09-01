from itertools import product
from math import sqrt, ceil
from multiprocessing import Pool, Lock
from time import sleep
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from numpy import float64

import config
from dynamics import (
    get_initial_state,
    get_state_altitude,
    state_integrator,
    MASS,
)
from type_definitions import Vector3D


def test_orbit(altitude: float64, dt: float64 = 0.1):
    r = altitude + config.MOON_RADIUS
    v = sqrt(config.GRAVITATIONAL_CONSTANT * config.MOON_MASS / r)

    state = get_initial_state(altitude, np.array([0.0, v, 0.0]), config.FULL_MASS)
    period = 2 * np.pi * r / v

    orbits = 1
    total_time = orbits * period
    t = 0

    # Time, state, altitude

    data = np.zeros((ceil(total_time / dt) + 1, 9), dtype=object)
    data[0] = np.concatenate([[t], state, [get_state_altitude(state)]])
    i = 1

    print(f"Initial velocity: {v:.5f}m/s")
    print(f"Period time: {period:.2f}s")
    print(f"Simulating: {total_time:.2f}s")
    sleep(1)

    while t <= total_time:
        next_state = state_integrator(state, np.array([0.0, 0.0, 0.0]), dt)

        state = next_state

        data[i] = np.concatenate([[t], state, [get_state_altitude(state)]])

        t += dt
        i += 1

        if i % 100 == 0:
            print(
                f"{100*(t/total_time):.2f}% - {t:.2f}s - {get_state_altitude(state):.4f}m"
            )

    return state, data


def test_orbit_prograde_thrust(altitude: float64, thrust: float64, dt: float64 = 0.1):
    r = altitude + config.MOON_RADIUS
    v = sqrt(config.GRAVITATIONAL_CONSTANT * config.MOON_MASS / r)

    state = get_initial_state(altitude, np.array([0.0, v, 0.0]), config.FULL_MASS)
    period = 2 * np.pi * r / v

    orbits = 4
    total_time = orbits * period
    t = 0

    # Time, state, altitude

    data = np.zeros((ceil(total_time / dt) + 1, 9), dtype=object)
    data[0] = np.concatenate([[t], state, [get_state_altitude(state)]])
    i = 1

    print(f"Initial velocity: {v:.5f}m/s")
    print(f"Period time: {period:.2f}s")
    print(f"Simulating: {total_time:.2f}s")
    sleep(1)

    while t <= total_time:
        velocity = state[3:6]
        control_dir = velocity / np.linalg.norm(velocity)
        control = control_dir * thrust

        next_state = state_integrator(state, control, dt)

        state = next_state

        data[i] = np.concatenate([[t], state, [get_state_altitude(state)]])

        t += dt
        i += 1

        if i % 100 == 0:
            print(
                f"{100*(t/total_time):.2f}% - {t:.2f}s - {get_state_altitude(state):.4f}m - {state[MASS]:.1f}kg"
            )

    return state, data


def show_3d(data):
    t = data[..., 0].astype(np.float64)
    x = data[..., 1].astype(np.float64)
    y = data[..., 2].astype(np.float64)
    z = data[..., 3].astype(np.float64)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    scatter = ax.scatter(x, y, z, c=t, cmap="viridis", s=6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(scatter, label="Time")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1000:g} km"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1000:g} km"))
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1000:g} km"))

    # Add moon
    r = config.MOON_RADIUS

    u = np.linspace(0, 2 * np.pi, 300)
    v = np.linspace(0, np.pi, 300)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color="b", alpha=0.3, edgecolor="navy", linewidth=0.3)

    ax.set_aspect("equal")

    plt.show()


def show_3d_plotly(data):
    import plotly.graph_objects as go

    data = data[data[:, 0] != -1]

    t = data[:, 0].astype(np.float64)
    x = data[:, 1].astype(np.float64)
    y = data[:, 2].astype(np.float64)
    z = data[:, 3].astype(np.float64)
    alt = data[:, 8].astype(np.float64)
    vel = data[:, 9]
    angle = data[:, 10].astype(np.float64)
    thrust = data[:, 11]

    df = pd.DataFrame(
        {
            "X": x,
            "Y": y,
            "Z": z,
            "Time": [round(d, 2) for d in t],
            "Alt": [round(d) for d in alt],
            "Vel": [round(float64(np.linalg.norm(d))) for d in vel],
            "Pitch": [round(d, 2) for d in angle],
            "RawThrX": [t[0] for t in thrust],
            "RawThrY": [t[1] for t in thrust],
            "RawThrZ": [t[2] for t in thrust],
        }
    )

    quiver_df = df.iloc[::40]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=3,
                color=t,
                colorscale="Viridis",
                colorbar=dict(title="Time"),
                showscale=True,
                opacity=0.95,
            ),
            customdata=df[["Time", "Alt", "Vel", "Pitch"]],
            hovertemplate=(
                "<b>Time:</b> %{customdata[0]}s<br>"
                "<b>Altitude:</b> %{customdata[1]}m<br>"
                "<b>Velocity:</b> %{customdata[2]}m/s<br>"
                "<b>Pitch:</b> %{customdata[3]}rad<br>"
            ),
            name="Trajectory",
        )
    )

    fig.add_cone(
        x=quiver_df["X"],
        y=quiver_df["Y"],
        z=quiver_df["Z"],
        u=quiver_df["RawThrX"],
        v=quiver_df["RawThrY"],
        w=quiver_df["RawThrZ"],
        sizemode="absolute",
        sizeref=100000,
        anchor="tail",
        hoverinfo="skip",
    )

    r = config.MOON_RADIUS

    n_lat = 360
    n_lon = 360

    for i in range(n_lat + 1):
        v_i = i * np.pi / n_lat
        theta = np.linspace(0, 2 * np.pi, 100)
        x_lat = r * np.cos(theta) * np.sin(v_i)
        y_lat = r * np.sin(theta) * np.sin(v_i)
        z_lat = r * np.ones_like(theta) * np.cos(v_i)

        fig.add_trace(
            go.Scatter3d(
                x=x_lat,
                y=y_lat,
                z=z_lat,
                mode="lines",
                line=dict(color="navy", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for i in range(n_lon + 1):
        u_i = i * 2 * np.pi / n_lon
        phi = np.linspace(0, np.pi, 100)
        x_lon = r * np.cos(u_i) * np.sin(phi)
        y_lon = r * np.sin(u_i) * np.sin(phi)
        z_lon = r * np.ones_like(phi) * np.cos(phi)

        fig.add_trace(
            go.Scatter3d(
                x=x_lon,
                y=y_lon,
                z=z_lon,
                mode="lines",
                line=dict(color="navy", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    ranges = [np.max(d) - np.min(d) for d in [x, y, z]]
    max_range = np.max(ranges)

    around = max_range * 1.2

    midpoints = [(np.max(d) + np.min(d)) / 2 for d in [x, y, z]]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[midpoints[0] + around, midpoints[0] - around]),
            yaxis=dict(title="Y", range=[midpoints[1] + around, midpoints[1] - around]),
            zaxis=dict(title="Z", range=[midpoints[2] + around, midpoints[2] - around]),
            aspectmode="data",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=-0.05, y=0.05, z=-0.8),
                up=dict(x=-1, y=0, z=0),
                center=dict(x=-0.02, y=0, z=0),
            ),
        ),
        autosize=True,
        showlegend=True,
    )

    fig.show()


def test_descent_pitch_law(
    altitude: float64,
    velocity: Vector3D,
    control_law: Callable[[float64], float64],
    dt: float64 = 0.1,
    end_time: float64 = 2000,
):
    state = get_initial_state(altitude, velocity, config.FULL_MASS)

    last_altitude = altitude

    t = 0

    # Time, state, altitude, velocity, pitch, thrust
    data = np.zeros((ceil(end_time / dt) + 1, 12), dtype=object)

    for i in range(len(data)):
        data[i][0] = -1

    data[0] = np.concatenate(
        [
            np.array([t], dtype=object),
            state,
            np.array(
                [
                    get_state_altitude(state),
                    state[3:6],
                    -1,
                    np.array([0, 0, 0], dtype=object),
                ],
                dtype=object,
            ),
        ],
        dtype=object,
    )
    i = 1
    last_w = None
    switched_to_last = False

    while True:
        if get_state_altitude(state) > last_altitude:
            break
        last_altitude = get_state_altitude(state)

        if np.linalg.norm(state[3:6]) < 10:
            break

        if get_state_altitude(state) < 500:
            break

        if t > end_time:
            break

        if (
            state[MASS]
            < config.FULL_MASS - config.PROPELLANT_MASS + config.LEAVE_FOR_LAND
        ):
            break

        theta = control_law(t)
        phi = -theta - np.pi / 2

        p = state[0:3]  # position
        a = -p
        b = state[3:6]  # velocity

        u = a / np.linalg.norm(a)  # unit vector of a

        c = np.vdot(b, u)
        proj = u * c  # Amount to shift velocity vector

        v = b - proj

        v_norm = np.linalg.norm(v)

        if (v_norm < 20 and (last_w is not None)) or switched_to_last:
            switched_to_last = True
            w = last_w
        else:
            w = v / np.linalg.norm(
                v
            )  # Perp. vector to ground, opposite direction of velocity
            last_w = w

        thrust_vector = np.cos(phi) * u + np.sin(phi) * w
        thrust_vector = thrust_vector / np.linalg.norm(thrust_vector)

        thrust_vector *= 0 if (theta == -100) else config.ENGINE_FORCE

        new_state = state_integrator(state, thrust_vector, dt)

        state = new_state

        data[i] = np.concatenate(
            [
                np.array([t], dtype=object),
                state,
                np.array(
                    [get_state_altitude(state), state[3:6], theta, thrust_vector],
                    dtype=object,
                ),
            ],
            dtype=object,
        )

        # if i % 100 == 99:
        # print(
        #    f"{t:5.2f}s: Altitude {get_state_altitude(state):.1f}m - Velocity {np.linalg.norm(b):.1f}m/s - Control at {theta:.2f} - Vector {np.linalg.norm(thrust_vector):.1f}"
        # )

        i += 1
        t += dt

    return state, data


def test_lin_tangent(
    altitude: float64,
    velocity: float64,
    a: float64,
    b: float64,
    start_time: float64,
    dt: float64,
    end_time: float64,
):
    def control_law(t: float64) -> float64:
        if t < start_time:
            return -100

        theta = np.arctan(a * (t - start_time) + b)
        return theta

    state, data = test_descent_pitch_law(altitude, velocity, control_law, dt, end_time)

    return state, data


def evaluate_params(params):
    global lock

    a, b, st = params
    final_state, data = test_lin_tangent(
        110000, np.array([0.0, 1400.0, 0.0]), a, b, st, 0.2, 1200
    )
    velocity = np.linalg.norm(final_state[3:6])
    alt = get_state_altitude(final_state)

    return {
        "a": a,
        "b": b,
        "st": st,
        "final_state": final_state,
        "data": data,
        "velocity": velocity,
        "alt": alt,
    }


if __name__ == "__main__":
    param_grid = list(
        product(
            np.linspace(0.0015, 0.0018, 5),
            np.linspace(0.004, 0.008, 5),
            np.linspace(400, 600, 5),
        )
    )

    print(f"Total combinations to evaluate: {len(param_grid)}")

    best_velocity = 100000
    best_state = None
    best_data = None
    best_args = [0, 0, 0]

    i = 1

    with Pool(processes=12) as pool:
        for result in pool.imap_unordered(evaluate_params, param_grid):
            print(
                f"{i:5.0f}: Tried {result['a']:.6f} {result['b']:.6f} {result['st']:.6f}s: "
                f"{result['alt']:.1f}m, {result['velocity']:.1f}m/s"
            )
            i += 1

            if result["alt"] > 2000:
                continue

            if result["velocity"] < best_velocity:
                best_velocity = result["velocity"]
                best_data = result["data"]
                best_state = result["final_state"]
                best_args = [result["a"], result["b"], result["st"]]

    print(
        f"Final altitude in {best_args[0]:.6f} {best_args[1]:.6f} {best_args[2]:.6f}s - {get_state_altitude(best_state):.3f}m - {np.linalg.norm(best_state[3:6]):.3f}m/s"
    )

    show_3d_plotly(best_data)
