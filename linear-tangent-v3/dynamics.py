# State vector: [x, y, z, v_x, v_y, v_z, mass]
import scipy
from numpy import float64

import config
from gravity import get_gravitational_force
from type_definitions import *
from config import *

P_X = 0
P_Y = 1
P_Z = 2
V_X = 3
V_Y = 4
V_Z = 5
MASS = 6


def get_state_gravity(state: State) -> Vector3D:
    return get_gravitational_force(
        state[MASS], MOON_MASS, np.array([0, 0, 0]), state[:3]
    )


def get_initial_state(altitude: float64, velocity: Vector3D, mass: float64) -> State:
    state = np.array([config.MOON_RADIUS + altitude, 0, 0, *velocity, mass])

    return state


def get_total_force(state: State, control: Vector3D) -> Vector3D:
    result = get_state_gravity(state) + control
    return result


def get_state_derivative(state: State, control: Vector3D) -> State:
    result = np.array([state[V_X], state[V_Y], state[V_Z], 0, 0, 0, 0])

    acceleration = get_total_force(state, control) / state[MASS]

    result[3] = acceleration[0]
    result[4] = acceleration[1]
    result[5] = acceleration[2]

    result[6] = -get_mass_flow(np.linalg.norm(control))

    return result


def state_integrator(state: State, control: Vector3D, dt: float64) -> State:
    def derivative(t, y):
        result = get_state_derivative(y, control)
        return result

    solution = scipy.integrate.solve_ivp(
        fun=derivative, t_span=(0.0, dt), y0=state, method="RK45"
    )
    new_state = solution.y[:, -1]
    return new_state


def get_mass_flow(thrust: float64) -> float64:
    g = 9.80665
    isp = config.ENGINE_ISP

    mass_flow = thrust / isp / g

    return mass_flow


def get_state_altitude(state: State) -> float64:
    r = np.linalg.norm(state[:3])
    return r - config.MOON_RADIUS
