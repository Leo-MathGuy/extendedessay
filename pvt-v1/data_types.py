import json
from typing import Literal, Self

import numpy as np
from scipy.constants import gravitational_constant, g

from scipy.linalg import norm

type Vector2D = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
type Vector3D = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
type Vector6D = np.ndarray[tuple[Literal[6]], np.dtype[np.float64]]


class WorldConfig:
    def __init__(
        self, moon_center: Vector3D, moon_mass: np.float64, moon_radius: np.float64
    ):
        self.moon_center: Vector3D = moon_center
        self.moon_mass: np.float64 = moon_mass
        self.moon_radius: np.float64 = moon_radius

    def to_json(self) -> str:
        return json.dumps(
            {
                "center": self.moon_center.tolist(),
                "mass": np.float64(self.moon_mass),
                "radius": np.float64(self.moon_radius),
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        data_dict = json.loads(json_str)
        return cls(
            moon_center=np.array(data_dict["center"], dtype=np.float64),
            moon_mass=np.float64(data_dict["mass"]),
            moon_radius=np.float64(data_dict["radius"]),
        )


class ShipConfig:
    def __init__(self, engine_thrust: np.float64, engine_isp: np.float64):
        self.engine_thrust: np.float64 = engine_thrust
        self.engine_isp: np.float64 = engine_isp

    @property
    def max_mass_flow(self) -> np.float64:
        return self.engine_thrust / (self.engine_isp * g)

    def get_mass_flow(self, thrust: np.float64) -> np.float64:
        return self.max_mass_flow * (thrust / self.engine_thrust)

    def to_json(self) -> str:
        return json.dumps(
            {
                "thrust": np.float64(self.engine_thrust),
                "isp": np.float64(self.engine_isp),
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        data_dict = json.loads(json_str)
        return cls(
            engine_thrust=np.float64(data_dict["thrust"]),
            engine_isp=np.float64(data_dict["isp"]),
        )


class State:
    def __init__(
        self,
        p_x: np.float64,
        p_y: np.float64,
        p_z: np.float64,
        v_x: np.float64,
        v_y: np.float64,
        v_z: np.float64,
        c_r: Vector3D,
        c_v: Vector3D,
        min_mass: np.float64,
        extra_mass: np.float64,
        extra_data=None,
    ):
        if extra_data is None:
            extra_data = dict()

        self.p_x: np.float64 = p_x
        self.p_y: np.float64 = p_y
        self.p_z: np.float64 = p_z
        self.v_x: np.float64 = v_x
        self.v_y: np.float64 = v_y
        self.v_z: np.float64 = v_z

        """Minimum mass"""
        self.min_mass: np.float64 = min_mass

        """Usable fuel mass"""
        self.extra_mass: np.float64 = extra_mass

        self.extra_data = extra_data

        self.c_r: Vector3D = c_r
        self.c_v: Vector3D = c_v

    @classmethod
    def from_vector(
        cls,
        state_vector: Vector6D,
        costate_vector: Vector6D,
        min_mass: np.float64,
        extra_mass: np.float64,
        extra_data: dict = None,
    ) -> Self:
        return State(
            *(state_vector.tolist()),
            costate_vector[:3],
            costate_vector[3:],
            min_mass,
            extra_mass,
            extra_data,
        )

    @property
    def x(self) -> Vector6D:
        """State vector"""
        return np.array(
            [self.p_x, self.p_y, self.p_z, self.v_x, self.v_y, self.v_z], np.float64
        )

    @property
    def pos(self) -> Vector3D:
        """Position vector"""
        return np.array([self.p_x, self.p_y, self.p_z], np.float64)

    @property
    def vel(self) -> Vector3D:
        """Velocity vector"""
        return np.array([self.v_x, self.v_y, self.v_z], np.float64)

    def dx_dt(self, world_config: WorldConfig, thrust: Vector3D) -> Vector6D:
        """State vector derivative. Provide the force of the thrust in Newtons"""
        ret = np.ndarray((6,))
        ret[0:3] = [self.v_x, self.v_y, self.v_z]

        offset: Vector3D = self.pos - world_config.moon_center
        R = norm(offset)

        g_magnitude = gravitational_constant * world_config.moon_mass / R**2
        acc_direction = -offset / norm(offset)
        acceleration = acc_direction * g_magnitude
        acceleration += thrust / self.mass

        ret[3:6] = acceleration

        return ret

    @property
    def mass(self):
        """Get total current mass"""
        return self.min_mass + self.extra_mass

    def altitude(self, world_config: WorldConfig) -> np.float64:
        return norm(self.pos - world_config.moon_center) - world_config.moon_radius

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        data_dict = json.loads(json_str)
        pos = data_dict["pos"]
        vel = data_dict["vel"]
        costate = data_dict["costate"]
        min_mass = np.float64(data_dict["min_mass"])
        extra_mass = np.float64(data_dict["extra_mass"])
        extra_data = data_dict["extra_data"]

        return cls.from_vector(
            np.array([*pos, *vel], dtype=np.float64), min_mass, extra_mass, extra_data
        )

    @property
    def costate(self) -> Vector6D:
        return np.array([*self.c_r, *self.c_v], np.float64)

    def to_json(self) -> str:
        pos = self.pos.tolist()
        vel = self.vel.tolist()
        min_mass = self.min_mass
        extra_mass = self.extra_mass
        extra_data = self.extra_data
        costate = self.costate.tolist()

        d = {
            "pos": pos,
            "vel": vel,
            "min_mass": min_mass,
            "extra_mass": extra_mass,
            "extra_data": extra_data,
            "costate": costate,
        }

        return json.dumps(d)


class TrajectoryHistory:
    def __init__(self, length: int, extra_data: dict = None):
        self.length = length
        self.extra_data = extra_data
        self.i: int = 0

        # Time
        # State
        # Derivatives
        # Control

        self.array = np.ndarray((length, 3), dtype=object)

    def append(
        self,
        time: float,
        state: State,
        derviatives: list[float] | None,
        control: list[float] | None,
    ):
        self.array[self.i] = np.array([time, State, derviatives, control], dtype=object)
        self.i += 1

    def times(self) -> np.ndarray:
        return self.array[: self.i - 1, 0]

    def states(self) -> np.ndarray:
        return self.array[: self.i - 1, 1]

    def derivatives(self) -> np.ndarray:
        return self.array[: self.i - 1, 2]

    def controls(self) -> np.ndarray:
        return self.array[: self.i - 1, 3]

    @classmethod
    def from_lists(
        cls,
        curi: int,
        extra_data: dict | None,
        times: np.ndarray,
        states: np.ndarray,
        derivatives: list[list[float]],
        controls: list[list[float]],
    ) -> Self:
        result = cls(curi, extra_data)

        for i, time in enumerate(times):
            result.array[i] = np.array(
                [time, states[i], derivatives[i], controls[i]], dtype=object
            )

        return result

    def to_json(self) -> str:
        times = self.times().tolist()
        states: list[State] = self.states().tolist()
        derviatives: list[list[float]] = self.derivatives().tolist()
        controls: list[list[float]] = self.controls().tolist()

        return json.dumps(
            {
                "extra_data": self.extra_data,
                "i": self.i - 1,
                "times": times,
                "states": [x.to_json() for x in states],
                "derivatives": [x or [0xDEADBEEF] * 6 for x in derviatives],
                "controls": [x or [0xDEADBEEF] * 3 for x in controls],
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        data_dict = json.loads(json_str)

        times = data_dict["times"]
        states = data_dict["states"]
        states = np.array([[State.from_json(x) for x in states]])
        derivatives = data_dict["derivatives"]
        derivatives = [x if x != [0xDEADBEEF] * 6 else None for x in derivatives]
        controls = data_dict["controls"]
        controls = [x if x != [0xDEADBEEF] * 6 else None for x in controls]

        return cls.from_lists(
            data_dict["i"],
            data_dict["extra_data"],
            times,
            states,
            derivatives,
            controls,
        )
