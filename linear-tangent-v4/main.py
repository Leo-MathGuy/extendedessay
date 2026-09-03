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

        l.info("Starting simulation")
        l.info(
            " - Altitude {} ; Velocity {} ; Simulating {} seconds at dt {}".format(
                round(altitude / 1000, 3) + "km",
                round(velocity) + "m/s",
                round(time_f / 60, 1) + "min",
                dt + "s",
            )
        )
        data = dynamics_examples.simulate_dynamics(altitude, velocity, time_f, dt, 2)
        data = data[data[:, 0] != -1]
        last_state = data[-1]

        l.info("Finished simulation")
        l.info(
            " - Altitude {} ; Velocity {} ; Simulating {} seconds at dt {}".format(
                round(altitude / 1000, 3) + "km",
                round(velocity) + "m/s",
                round(time_f / 60, 1) + "min",
                dt + "s",
            )
        )
