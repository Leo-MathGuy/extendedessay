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

def integrate_pvt_state(state: State, world_conf: WorldConfig) -> float:
