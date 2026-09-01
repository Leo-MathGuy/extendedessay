from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.constants import gravitational_constant

from scipy.linalg import norm

type Vector2D = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
type Vector3D = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
type Vector6D = np.ndarray[tuple[Literal[6]], np.dtype[np.float64]]


class WorldConfig:
    def __init__(self, moon_center: Vector3D, moon_mass: np.float64, moon_radius: np.float64):
        self.moon_center: Vector3D = moon_center
        self.moon_mass: np.float64 = moon_mass
        self.moon_radius: np.float64 = moon_radius

class State:
    def __init__(self, p_x: np.float64, p_y: np.float64, p_z: np.float64, v_x: np.float64, v_y: np.float64, v_z: np.float64, dry_mass: np.float64, fuel_mass: np.float64):
        self.p_x: np.float64 = p_x
        self.p_y: np.float64 = p_y
        self.p_z: np.float64 = p_z
        self.v_x: np.float64 = v_x
        self.v_y: np.float64 = v_y
        self.v_z: np.float64 = v_z

        self.dry_mass: np.float64 = dry_mass
        self.fuel_mass: np.float64 = fuel_mass


    @property
    def x(self) -> Vector6D:
        """State vector"""
        return np.array([self.p_x, self.p_y, self.p_z, self.v_x, self.v_y, self.v_z], np.float64)

    @property
    def pos(self) -> Vector3D:
        """Position vector"""
        return np.array([self.p_x, self.p_y, self.p_z], np.float64)

    @property
    def vel(self) -> Vector3D:
        """Velocity vector"""
        return np.array([self.v_x, self.v_y, self.v_z], np.float64)

    def dx_dt(self, world_config: WorldConfig, thrust: Vector3D) -> Vector6D:
        """State vector derivative. Provide the force of the thrust"""
        ret = np.ndarray((6,))
        ret[0:3] = [self.v_x, self.v_y, self.v_z]

        offset: Vector3D = self.pos - world_config.moon_center
        R = norm(offset)

        g_magnitude = gravitational_constant * world_config.moon_mass / R^2
        acc_direction = -offset / norm(offset)
        acceleration = acc_direction * g_magnitude
        acceleration += thrust

        ret[3:6] = acceleration

        return ret

    @property
    def get_mass(self):
        return self.dry_mass + self.fuel_mass
