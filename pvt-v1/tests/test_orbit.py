import logging

import numpy as np
from scipy.constants import gravitational_constant

import constants
from constants import MOON_RADIUS
from dynamics_examples import orbit_test

import logs
import pint

ureg = pint.UnitRegistry()

l = logging.getLogger("l")


def test_orbit_circular():
    alt = 100 * 1000
    r = MOON_RADIUS + alt
    g = gravitational_constant * constants.MOON_MASS / (r**2)
    velocity = np.sqrt(g * r)

    period = 2 * np.pi * r / velocity

    l.info("Testing 20 orbits at 100km")
    data, world, ship = orbit_test(
        altitude=alt, velocity=velocity, time_f=period * 20, dt=25, log=0
    )
    final_state = data[-1][1]
    final_alt = final_state.altitude(world)

    l.info(f"Final altitude deviated by {(final_alt - alt)*ureg.meter:.3g~#}")
    assert abs(final_alt - alt) < 0.05


def test_orbit_circular_2():
    alt = 10 * 1000
    r = MOON_RADIUS + alt
    g = gravitational_constant * constants.MOON_MASS / (r**2)
    velocity = np.sqrt(g * r)

    period = 2 * np.pi * r / velocity

    l.info("Testing 200 lower resolution orbits at 10km")
    data, world, ship = orbit_test(
        altitude=alt, velocity=velocity, time_f=period * 200, dt=100, log=0
    )
    final_state = data[-1][1]
    final_alt = final_state.altitude(world)

    l.info(f"Final altitude deviated by {(final_alt - alt)*ureg.meter:.3g~#}")
    assert abs(final_alt - alt) < 1
