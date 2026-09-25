import logging

import numpy as np
from scipy import integrate
from scipy.constants import gravitational_constant
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


def integrate_state_rk4(state, world_cfg, ship_cfg, thrust, dt):
    if norm(thrust) < 0.0001:
        mu = gravitational_constant * world_cfg.moon_mass
        r_new, v_new = kepler_propagate(state.pos, state.vel, dt, mu)
        return State.from_vector(
            np.concatenate([r_new, v_new]),
            state.min_mass,
            state.extra_mass,
            extra_data=state.extra_data,
        )

    y0 = state.x
    m0 = state.extra_mass
    mass_flow = ship_cfg.get_mass_flow(norm(thrust))

    def f(y, mass):
        s = State.from_vector(y, state.min_mass, mass)
        return s.dx_dt(world_cfg, thrust)

    k1 = f(y0, m0)
    k2 = f(y0 + 0.5 * dt * k1, m0 - 0.5 * dt * mass_flow)
    k3 = f(y0 + 0.5 * dt * k2, m0 - 0.5 * dt * mass_flow)
    k4 = f(y0 + dt * k3, m0 - dt * mass_flow)

    y_new = y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    mass_new = m0 - mass_flow * dt
    return State.from_vector(y_new, state.min_mass, max(mass_new, 0.0))


def kepler_propagate(r0, v0, dt, mu):
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r0n = np.linalg.norm(r0)
    v0n = np.linalg.norm(v0)
    vr0 = np.dot(r0, v0) / r0n
    alpha = 2.0 / r0n - v0n * v0n / mu
    if alpha > 1e-12:
        chi = np.sqrt(mu) * dt * alpha
    elif alpha < -1e-12:
        chi = (
            np.sign(dt)
            * np.sqrt(-1.0 / alpha)
            * np.log(
                (-2.0 * mu * alpha * dt)
                / (vr0 + np.sign(dt) * np.sqrt(-mu * alpha) * (1.0 - r0n * alpha))
            )
        )
    else:
        chi = np.sqrt(mu) * dt / r0n
    for _ in range(50):
        z = alpha * chi * chi
        if z > 1e-8:
            sz = np.sqrt(z)
            C = (1.0 - np.cos(sz)) / z
            S = (sz - np.sin(sz)) / (sz**3)
        elif z < -1e-8:
            sz = np.sqrt(-z)
            C = (1.0 - np.cosh(sz)) / z
            S = (np.sinh(sz) - sz) / (sz**3)
        else:
            C = 0.5 - z / 24.0
            S = 1.0 / 6.0 - z / 120.0
        F = (
            r0n * vr0 / np.sqrt(mu) * chi * chi * C
            + (1.0 - alpha * r0n) * chi**3 * S
            + r0n * chi
            - np.sqrt(mu) * dt
        )
        dF = (
            r0n * vr0 / np.sqrt(mu) * chi * (1.0 - z * S)
            + (1.0 - alpha * r0n) * chi * chi * C
            + r0n
        )
        dchi = F / dF
        chi -= dchi
        if abs(dchi) < 1e-12:
            break
    f = 1.0 - (chi * chi / r0n) * C
    g = dt - (chi**3 / np.sqrt(mu)) * S
    r = f * r0 + g * v0
    rn = np.linalg.norm(r)
    fdot = (np.sqrt(mu) / (rn * r0n)) * chi * (z * S - 1.0)
    gdot = 1.0 - (chi * chi / rn) * C
    v = fdot * r0 + gdot * v0
    return r, v


def integrate_state_rk4_thrusted(
    state, world_cfg, ship_cfg, thrust_fn, t0, dt, ignition_time
):
    mu = gravitational_constant * world_cfg.moon_mass

    def thrust_at(t_abs, mass):
        if t_abs < ignition_time or mass <= 0.0:
            return np.zeros(3)
        return np.asarray(thrust_fn(t_abs), dtype=float)

    def rk4_substep(st, t_start, h):
        y0 = st.x.copy()
        m0 = float(st.extra_mass)
        thr0 = thrust_at(t_start, m0)
        if norm(thr0) < 1e-6:
            r_new, v_new = kepler_propagate(st.pos, st.vel, h, mu)
            return State.from_vector(
                np.concatenate([r_new, v_new]),
                st.min_mass,
                m0,
                extra_data=st.extra_data,
            )
        mf = ship_cfg.get_mass_flow(norm(thr0))

        def acc(y, mass, t_abs):
            thr = thrust_at(t_abs, mass)
            s = State.from_vector(y, st.min_mass, mass)
            return s.dx_dt(world_cfg, thr)

        k1 = acc(y0, m0, t_start)
        k2 = acc(y0 + 0.5 * h * k1, m0 - 0.5 * h * mf, t_start + 0.5 * h)
        k3 = acc(y0 + 0.5 * h * k2, m0 - 0.5 * h * mf, t_start + 0.5 * h)
        k4 = acc(y0 + h * k3, m0 - h * mf, t_start + h)
        y_new = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        m_new = max(m0 - mf * h, 0.0)
        return State.from_vector(y_new, st.min_mass, m_new, extra_data=st.extra_data)

    events = []
    if t0 < ignition_time < t0 + dt:
        events.append(ignition_time - t0)
    thr_now = thrust_at(t0 + 1e-9, state.extra_mass)
    if norm(thr_now) > 1e-6 and state.extra_mass > 0.0:
        mf = ship_cfg.get_mass_flow(norm(thr_now))
        if mf > 1e-12:
            t_dep = state.extra_mass / mf
            if 0.0 < t_dep < dt:
                events.append(t_dep)
    events = sorted(set([0.0] + events + [dt]))

    cur = state
    for i in range(len(events) - 1):
        h = events[i + 1] - events[i]
        if h <= 0.0:
            continue
        cur = rk4_substep(cur, t0 + events[i], h)
    return cur
