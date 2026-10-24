import numpy as np
import pandas as pd
from numpy import float64
from scipy.linalg import norm

import constants
from data_types import State, WorldConfig, ShipConfig


def show_3d_plotly_trajectory(
    data: list[tuple[float64, State]], world: WorldConfig, ship: ShipConfig
):
    import plotly.graph_objects as go

    df = pd.DataFrame(
        {
            "X": [d[1].p_x for d in data],
            "Y": [d[1].p_y for d in data],
            "Z": [d[1].p_z for d in data],
            "Time": [d[0] for d in data],
            "Alt": [d[1].altitude(world) for d in data],
            "Vel": [norm(d[1].vel) for d in data],
        }
    )

    quiver_df = df.iloc[::40]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=df["X"].tolist(),
            y=df["Y"].tolist(),
            z=df["Z"].tolist(),
            mode="markers",
            marker=dict(
                size=3,
                color=df["Time"].tolist(),
                colorscale="Viridis",
                colorbar=dict(title="Time"),
                showscale=True,
                opacity=0.95,
            ),
            customdata=df[["Time", "Alt", "Vel"]],
            hovertemplate=(
                "<b>Time:</b> %{customdata[0]}s<br>"
                "<b>Altitude:</b> %{customdata[1]}m<br>"
                "<b>Velocity:</b> %{customdata[2]}m/s<br>"
                "<b>Pitch:</b> %{customdata[3]}rad<br>"
            ),
            name="Trajectory",
        )
    )

    # fig.add_cone(
    #     x=quiver_df["X"],
    #     y=quiver_df["Y"],
    #     z=quiver_df["Z"],
    #     u=quiver_df["RawThrX"],
    #     v=quiver_df["RawThrY"],
    #     w=quiver_df["RawThrZ"],
    #     sizemode="absolute",
    #     sizeref=100000,
    #     anchor="tail",
    #     hoverinfo="skip",
    # )

    r = constants.MOON_RADIUS

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

    ranges = [
        np.max(d) - np.min(d)
        for d in [df["X"].tolist(), df["Y"].tolist(), df["Z"].tolist()]
    ]
    max_range = np.max(ranges)

    around = max_range * 1.2

    midpoints = [
        (np.max(d) + np.min(d)) / 2
        for d in [df["X"].tolist(), df["Y"].tolist(), df["Z"].tolist()]
    ]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[midpoints[0] + around, midpoints[0] - around]),
            yaxis=dict(title="Y", range=[midpoints[1] + around, midpoints[1] - around]),
            zaxis=dict(title="Z", range=[midpoints[2] + around, midpoints[2] - around]),
            aspectmode="cube",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=-0.05, y=0.05, z=-0.8),
                up=dict(x=-1, y=0, z=0),
                center=dict(x=-0.02, y=0, z=0),
            ),
        ),
        showlegend=True,
    )

    fig.show()


def show_3d_plotly_trajectory_lintan(
    data: list[tuple[float64, State]], world: WorldConfig, ship: ShipConfig
):
    import plotly.graph_objects as go

    engine_start = data[0][1].extra_data.get("start_time", 0)

    thrusts = [
        (
            np.array(d[1].extra_data["thrust"], np.float64)
            if d[0] > engine_start
            else np.array([0, 0, 0])
        )
        for d in data
    ]

    df = pd.DataFrame(
        {
            "X": [d[1].p_x for d in data],
            "Y": [d[1].p_y for d in data],
            "Z": [d[1].p_z for d in data],
            "Time": [round(d[0], 2) for d in data],
            "Alt": [round(d[1].altitude(world), 2) for d in data],
            "Vel": [round(norm(d[1].vel), 2) for d in data],
            "Pitch": [round(d[1].extra_data["angle"], 2) for d in data],
            "RawThrX": [d[0] for d in thrusts],
            "RawThrY": [d[1] for d in thrusts],
            "RawThrZ": [d[2] for d in thrusts],
        }
    )

    quiver_df = df.iloc[::40]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=df["X"].tolist(),
            y=df["Y"].tolist(),
            z=df["Z"].tolist(),
            mode="markers",
            marker=dict(
                size=3,
                color=df["Time"].tolist(),
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
                "<b>Pitch:</b> %{customdata[3]}deg<br>"
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

    r = constants.MOON_RADIUS

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

    ranges = [
        np.max(d) - np.min(d)
        for d in [df["X"].tolist(), df["Y"].tolist(), df["Z"].tolist()]
    ]
    max_range = np.max(ranges)

    around = max_range * 0.6

    midpoints = [
        (np.max(d) + np.min(d)) / 2
        for d in [df["X"].tolist(), df["Y"].tolist(), df["Z"].tolist()]
    ]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[midpoints[0] + around, midpoints[0] - around]),
            yaxis=dict(title="Y", range=[midpoints[1] + around, midpoints[1] - around]),
            zaxis=dict(title="Z", range=[midpoints[2] + around, midpoints[2] - around]),
            aspectmode="cube",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=-0.05, y=0.05, z=-0.8),
                up=dict(x=-1, y=0, z=0),
                center=dict(x=-0.02, y=0, z=0),
            ),
        ),
        showlegend=True,
    )

    fig.show()


def show_penalty_mesh(a_list, b_list, penalties):
    import plotly.graph_objects as go

    A, B = np.meshgrid(a_list, b_list, indexing="ij")

    penalty_plot = penalties.astype(float).copy()
    penalty_plot[penalty_plot == -1] = np.nan

    fig = go.Figure(
        data=[
            go.Surface(
                x=A,
                y=B,
                z=penalty_plot,
                colorscale="Viridis",
                colorbar=dict(title="Penalty"),
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="limegreen",
                        project_z=True,
                    )
                ),
                connectgaps=False,
            )
        ]
    )

    fig.update_layout(
        title="Penalty Gradient Plane",
        scene=dict(
            xaxis_title="a",
            yaxis_title="b",
            zaxis_title="Penalty",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        width=900,
        height=700,
    )

    fig.show()
