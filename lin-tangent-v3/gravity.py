import numpy as np

import config

from type_definitions import Vector3D


def get_gravitational_force(
    m1: np.float64, m2: np.float64, p_object: Vector3D, p_other: Vector3D
) -> Vector3D:
    r = np.linalg.norm(p_object - p_other)
    magnitude = config.GRAVITATIONAL_CONSTANT * m1 * m2 / (r**2)

    result = (p_object - p_other) / r * magnitude

    return result
