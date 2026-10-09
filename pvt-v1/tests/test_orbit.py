import numpy as np
from scipy.constants import gravitational_constant

import constants
from constants import MOON_RADIUS
from dynamics_examples import orbit_test


def test_orbit_circular():
    alt = 100 * 1000
    r = MOON_RADIUS + alt
    g = gravitational_constant * constants.MOON_MASS / (r**2)
    velocity = np.sqrt(g * r)

    period = 2 * np.pi * r / velocity

    print("Testing 20 orbits at 100km")
    data, world, ship = orbit_test(
        altitude=alt, velocity=velocity, time_f=period * 20, dt=25, log=1
    )
    final_state = data[-1][1]
    final_alt = final_state.altitude(world)

    print(f"Final altitude deviated by {round(final_alt - alt, 6)}")
    assert abs(final_alt - alt) < 0.05


def test_orbit_circular_2():
    alt = 10 * 1000
    r = MOON_RADIUS + alt
    g = gravitational_constant * constants.MOON_MASS / (r**2)
    velocity = np.sqrt(g * r)

    period = 2 * np.pi * r / velocity

    print("Testing 200 lower resolution orbits at 10km")
    data, world, ship = orbit_test(
        altitude=alt, velocity=velocity, time_f=period * 200, dt=100, log=1
    )
    final_state = data[-1][1]
    final_alt = final_state.altitude(world)

    print(f"Final altitude deviated by {round(final_alt - alt, 6)}")
    assert abs(final_alt - alt) < 1
