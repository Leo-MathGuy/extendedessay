from concurrent.futures import as_completed, ThreadPoolExecutor
from time import sleep
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
import scipy.integrate as spi
from black import datetime
from matplotlib import pyplot as plt

import conf

Vector2D = Annotated[npt.NDArray[np.float64], Literal[1, 2]]


def plot(history):
    (
        time,
        mass,
        altitude,
        velocity,
        angle_rad,
        position,
        velocity_vector,
        angle_global_rad,
        control_v,
        fuel,
    ) = zip(*history)

    time = np.array(time)
    mass = np.array(mass)
    altitude = np.array(altitude)
    velocity = np.array(velocity)
    angle_rad = np.array(angle_rad)
    position = np.array(position)
    velocity_vector = np.array(velocity_vector)
    angle_global_rad = np.array(angle_global_rad)
    control_v = np.array(control_v)
    fuel = np.array(fuel)

    angle_deg = np.degrees(angle_rad)
    angle_global_deg = np.degrees(angle_global_rad)

    fig, axs = plt.subplots(2, 2, figsize=(10, 12), sharex=True)

    axs[0][0].plot(time, mass, "b-", linewidth=2)
    axs[0][0].set_ylabel("Mass (kg)")
    axs[0][0].grid(True)

    axs[0][1].plot(time, altitude, "g-", linewidth=2)
    axs[0][1].set_ylabel("Altitude (m)")
    axs[0][1].grid(True)

    axs[1][0].plot(time, angle_deg, "r-", linewidth=2)
    axs[1][0].set_ylabel("Control Angle (degrees)")
    axs[1][0].set_xlabel("Time (s)")
    axs[1][0].grid(True)

    axs[1][1].plot(time, velocity, "m-", linewidth=2)
    axs[1][1].set_ylabel("Velocity (m/s)")
    axs[1][0].set_xlabel("Time (s)")
    axs[1][1].grid(True)

    plt.suptitle("Optimizer", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


class State:
    def __init__(
        self, position: Vector2D, velocity: Vector2D, mass: float, propmass: float
    ):
        self.position = position
        self.velocity = velocity

        self.mass = mass
        self.propmass = propmass

        self.max_thrust = conf.ENGINE_FORCE
        self.isp = conf.ENGINE_ISP
        self.max_mass_flow = self.max_thrust / self.isp / 9.81

    def __repr__(self):
        return f"State(position={','.join([str(round(x, 2)) for x in self.position.tolist()])}, velocity={','.join([str(round(x, 2)) for x in self.velocity.tolist()])}, mass={self.mass}, propmass={self.propmass})"

    def calculate_gravity(self) -> Vector2D:
        gravity_magnitude = conf.MOON_GRAVITY * self.mass
        gravity_vector = (
            (-self.position) / np.linalg.norm(self.position) * gravity_magnitude
        )
        return gravity_vector

    def calculate_massflow(self, control: Vector2D) -> float:
        return self.max_mass_flow * float(np.linalg.norm(control) / self.max_thrust)

    def get_altitude(self) -> float:
        return float(np.linalg.norm(self.position)) - conf.MOON_RADIUS


def create_state() -> State:
    st = State(
        np.array([0, conf.MOON_RADIUS + conf.INITIAL_ALTITUDE]),
        np.array([conf.INITIAL_SPEED, 0]),
        conf.FULL_MASS,
        conf.PROPELLANT_MASS,
    )
    return st


state = create_state()


def state_integrator(
    input_state: State, control_input: Vector2D, delta_t: float
) -> State:
    state_vector = np.array(
        [
            *input_state.position,
            *input_state.velocity,
            input_state.mass,
            input_state.propmass,
            *control_input,
        ]
    )

    def derivative(t, y):
        pos = y[0:2]
        vel = y[2:4]
        mass = y[4]
        propmass = y[5]
        control = y[6:8]

        dydt = vel.copy()

        gravity_magnitude = conf.MOON_GRAVITY * mass
        gravity = (-np.array(pos)) / np.linalg.norm(np.array(pos)) * gravity_magnitude

        thrust = control
        acc = (gravity + thrust) / mass
        dydt = np.append(dydt, acc)

        mf = (conf.ENGINE_FORCE / conf.ENGINE_ISP / 9.81) * float(
            np.linalg.norm(np.array(control)) / conf.ENGINE_FORCE
        )
        dydt = np.append(dydt, [-mf, -mf, 0.0, 0.0])

        return dydt

    solution = spi.solve_ivp(
        fun=derivative,
        t_span=(0.0, delta_t),
        y0=state_vector,
        method="DOP853",
    )

    new_vector = solution.y[:, -1]

    new_state = State(
        position=new_vector[0:2],
        velocity=new_vector[2:4],
        mass=float(new_vector[4]),
        propmass=float(new_vector[5]),
    )

    return new_state


def compute_linear_tangent_control_angle(st: State, t: float, params) -> float:
    a, b = params[:2]
    start_time = params[2]

    theta_local = np.arctan(a * (t - start_time) + b)
    local_vertical = st.position / np.linalg.norm(st.position)

    vertical_global_angle = np.arctan2(local_vertical[1], local_vertical[0])
    theta_global = vertical_global_angle + theta_local

    return theta_global


def compute_linear_tangent_control(st: State, t: float, params) -> Vector2D:
    theta_global = compute_linear_tangent_control_angle(st, t, params)

    start_time = params[2]
    thrust_mag = st.max_thrust if t >= start_time else 0

    direction = np.array([np.cos(theta_global), np.sin(theta_global)])
    return thrust_mag * direction


def simulate_one_trajectory(
    initial_state: State,
    guidance_params: tuple,
    total_time: float,
    dt: float = 0.1,
):
    st = initial_state
    t = 0.0

    history = []

    i = 0
    last_alt = 1000000000
    while (
        t < total_time
        and st.propmass > conf.LEAVE_FOR_LAND
        and 500 < st.get_altitude() < last_alt
    ):
        last_alt = st.get_altitude()
        control = compute_linear_tangent_control(st, t, guidance_params)
        st = state_integrator(st, control, dt)
        t += dt

        theta_global = compute_linear_tangent_control_angle(st, t, guidance_params)

        local_vertical = st.position / np.linalg.norm(st.position)
        vertical_global_angle = np.arctan2(local_vertical[1], local_vertical[0])
        theta_local = theta_global - vertical_global_angle

        history.append(
            [
                t,
                st.propmass,
                st.get_altitude(),
                np.linalg.norm(st.velocity),
                theta_local,
                st.position,
                st.velocity,
                theta_global,
                control,
                st.propmass - conf.LEAVE_FOR_LAND,
            ]
        )
        i += 1

    propellant_used = initial_state.propmass - st.propmass
    return st, propellant_used, history, t


start = 550


def make_test_matrix(A_s, A_e, B_s, B_e, A_n, B_n):
    matrix = np.zeros((A_n, B_n, 7), dtype=object)

    for i, a in enumerate(np.arange(A_s, A_e, (A_e - A_s) / A_n)):
        for j, b in enumerate(np.arange(B_s, B_e, (B_e - B_s) / B_n)):
            print("Adding", round(a, 6), round(b, 6))
            matrix[i, j] = np.array([a, b, 0, 0, 0, None, None], dtype=object)

    return matrix


def process_parameter_pair(args):
    """
    Process a single (a, b) parameter pair.
    This function is designed to be called by threads.
    """
    i, j, b, start_time_initial, state = args

    print(f"Trying {round(b[0], 6)} {round(b[1], 6)}")

    start_time = start_time_initial
    time_gap = 1
    iteration = 0

    while True:
        iteration += 1
        result_state, pu, history, total_time = simulate_one_trajectory(
            state, (b[0], b[1], start_time), 500 + start_time, 0.1
        )

        f_center = b[0] * (total_time - start_time) + b[1]
        alt = history[-1][2]
        if alt > 700:
            break

        if 0 < f_center < 0.08:
            break

        result_state_l, pu_l, history_l, total_time_l = simulate_one_trajectory(
            state, (b[0], b[1], start_time), 500 + start_time - time_gap, 0.1
        )

        result_state_r, pu_r, history_r, total_time_r = simulate_one_trajectory(
            state, (b[0], b[1], start_time), 500 + start_time + time_gap, 0.1
        )

        f_left = b[0] * (total_time_l - (start_time - time_gap)) + b[1]
        f_right = b[0] * (total_time_r - (start_time + time_gap)) + b[1]
        f_prime = (f_right - f_left) / (time_gap * 2)
        t_new = start_time - 0.8 * f_center / f_prime
        start_time = t_new

        if start_time < 0:
            break

        if iteration > 50:
            break

    state_final = result_state

    return (
        i,
        j,
        np.array(
            [
                b[0],
                b[1],
                start_time,
                state_final.velocity,
                state_final.get_altitude(),
                state_final,
                history,
            ],
            dtype=object,
        ),
    )


def test_matrix(input_matrix, start_time_initial, state, max_workers=8):

    tasks = []
    for i, a in enumerate(input_matrix):
        for j, b in enumerate(a):
            tasks.append((i, j, b.copy(), start_time_initial, state))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_parameter_pair, task): task for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                i, j, result = future.result()
                input_matrix[i, j] = result
                print(
                    f"Finished {round(result[0], 6)} {round(result[1], 6)} "
                    f"alt {round(result[4], 6)} v {round(np.linalg.norm(result[3]), 2)}"
                )
            except Exception as e:
                print(f"Task failed for indices ({i}, {j}): {e}")

    return input_matrix


matrix_1 = make_test_matrix(-0.005, -0.01, 1.5, 5, 10, 8)
results = test_matrix(matrix_1, 800, state, max_workers=2)

best_result = np.array([])
best_velocity = 10000000
best_configs = []

for a in results:
    for b in a:
        if b[4] > 1500:
            continue
        if np.linalg.norm(b[3]) < best_velocity:
            best_velocity = np.linalg.norm(b[3])
            best_result = b
            best_configs = [
                b[0],
                b[1],
                b[2],
            ]

if len(best_result) == 0:
    print("failed to get best result")
    exit()

state_final = best_result[5]
history = best_result[6]


"""
best_configs = [-0.0003625, 2.055, 501.7]

state_final, prop, history = simulate_one_trajectory(
    state, best_configs, 500 + best_configs[2], 0.1
)
"""

print(state_final)
print(best_configs)

h = []

if len(history) > 0:
    for i, point in enumerate(history):
        if i > 0:
            if point[0] == 0:
                continue
        h.append(point)

(
    time,
    mass,
    altitude,
    velocity,
    angle_rad,
    position,
    velocity_vector,
    angle_global_rad,
    control_v,
    fuel,
) = zip(*history)

time = np.array(time)
mass = np.array(mass)
altitude = np.array(altitude)
velocity = np.array(velocity)
angle_rad = np.array(angle_rad)
position = np.array(position)
velocity_vector = np.array(velocity_vector)
angle_global_rad = np.array(angle_global_rad)
control_v = np.array(control_v)
fuel = np.array(fuel)

angle_deg = np.degrees(angle_rad)
angle_global_deg = np.degrees(angle_global_rad)

plot(history)