import logging

l = logging.getLogger("l")

import dynamics_examples

print("Select option")
print(" # 1 - Simulate altitude + velocity orbit")
print()
option = int(input("> "))

match option:
    case 1:
        altitude = float(input("Altitude: "))
        velocity = float(input("Velocity: "))
        time_f = float(input("Hours to simulate: ")) * 60 * 60
        dt = float(input("dt: "))
        print()

        dynamics_examples.test_orbit(altitude, velocity, time_f, dt)
