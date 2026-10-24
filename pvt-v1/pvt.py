from scipy import linalg

from data_types import State, Vector3D


def primer_vector(state: State) -> Vector3D:
    """It said the thing"""
    return -state.c_v


def thrust_directon(state: State) -> Vector3D:
    p = primer_vector(state)
    return p / linalg.norm(p)


def switching_function(state: State) -> Vector3D:
    return linalg.norm(primer_vector(state)) > 1

def cost_instant(state: State, control: Vector3D) -> float:
