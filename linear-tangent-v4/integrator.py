import logging

import numpy as np
from scipy import integrate
from scipy.integrate._ivp.ivp import OdeResult
from scipy.linalg import norm

from data_types import State, Vector3D, Vector6D, WorldConfig, ShipConfig

integration_modes = ["RK23", "RK45", "DOP853"]
using_mode = integration_modes[1]

l = logging.getLogger("l")


def integrate_state(
    state: State,
    world_config: WorldConfig,
    ship_config: ShipConfig,
    control: Vector3D,
    dt: np.float64,
    method: str = using_mode,
) -> State:
    state_vector: Vector6D = state.x

    mass_flow = ship_config.get_mass_flow(norm(control))

    def fun(t: float, y: Vector6D) -> Vector6D:
        expended_mass = mass_flow * t

        state_cur = State.from_vector(
            y, state.min_mass, state.extra_mass - expended_mass
        )
        return state_cur.dx_dt(world_config, control)

    solution: OdeResult[np.float64] = integrate.solve_ivp(
        fun=fun, y0=state.x, t_span=(0, dt), method=using_mode
    )

    return State.from_vector(
        solution.y[:, -1].copy(), state.min_mass, state.extra_mass - mass_flow * dt
    )
