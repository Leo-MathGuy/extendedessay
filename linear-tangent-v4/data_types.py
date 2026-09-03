import json
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
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


class ShipConfig:
    def __init__(self, engine_thrust: np.float64, engine_isp: np.float64):
        self.engine_thrust: np.float64 = engine_thrust
        self.engine_isp: np.float64 = engine_isp

    @property
    def max_mass_flow(self) -> np.float64:
        return self.engine_thrust / (self.engine_isp * g)

    def get_mass_flow(self, thrust: np.float64) -> np.float64:
        return ShipConfig.max_mass_flow * (thrust / self.engine_thrust)

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
        return cls(engine_thrust=data_dict["thrust"], engine_isp=data_dict["isp"])


class State:
    def __init__(
        self,
        p_x: np.float64,
        p_y: np.float64,
        p_z: np.float64,
        v_x: np.float64,
        v_y: np.float64,
        v_z: np.float64,
        min_mass: np.float64,
        extra_mass: np.float64,
    ):
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

    @classmethod
    def from_vector(
        cls, state_vector: Vector6D, min_mass: np.float64, extra_mass: np.float64
    ) -> Self:
        return State(*(state_vector.tolist()), min_mass, extra_mass)

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
        return norm(self.pos - world_config.moon_center)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        data_dict = json.loads(json_str)
        pos = data_dict["pos"]
        vel = data_dict["vel"]
        min_mass = np.float64(data_dict["min_mass"])
        extra_mass = np.float64(data_dict["extra_mass"])

        return cls.from_vector(
            np.array([*pos, *vel], dtype=np.float64), min_mass, extra_mass
        )

    def to_json(self) -> str:
        pos = self.pos.tolist()
        vel = self.vel.tolist()
        min_mass = self.min_mass
        extra_mass = self.extra_mass

        d = {
            "pos": pos,
            "vel": vel,
            "min_mass": min_mass,
            "extra_mass": extra_mass,
        }

        return json.dumps(d)
