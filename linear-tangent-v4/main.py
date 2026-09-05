import json
import logging
from random import randint

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.linalg import norm

import logs
import plotting
from data_types import State, ShipConfig, WorldConfig

l = logging.getLogger("l")

import dynamics_examples
from pathlib import Path

print("Select option")
print(" # 1 - Simulate altitude + velocity orbit")
print(" # 2 - Analyze 3d graph")
print()
option = int(input("> "))

OPTIONS_PATH = "data/options.json"
OUT_PATH = Path("data/out/")

options_file = open(OPTIONS_PATH, "r")
options_data = json.load(options_file)

match option:
    case 1:
        load = input("Load saved settings? y/n > ").lower() == "y"

        altitude = None
        velocity = None
        time_f = None
        dt = None

        if load:
            option_data = options_data.get("testorbit")
            if option_data is None:
                print("Failed to load settings")
                load = False
            else:
                for i, option in enumerate(option_data):
                    print(f" - Option {i} ({option.get('name')}):")
                    print(f" - - Alt {round(option.get('altitude')/1000, 2)}km")
                    print(f" - - Vel {round(option.get('velocity'),1)}m/s")
                    print(f" - - T_f {round(option.get('time_f')/60,1)} minutes")
                    print(f" - - dt {option.get('dt')}s")

                selected = int(input("Select option > "))
                data = option_data[selected]
                altitude = data["altitude"]
                velocity = data["velocity"]
                time_f = data["time_f"]
                dt = data["dt"]

        if not load:
            altitude = float(input("Altitude: "))
            velocity = float(input("Velocity: "))
            time_f = float(input("Hours to simulate: ")) * 60 * 60
            dt = float(input("dt: "))
            print()

        data, world, ship = dynamics_examples.orbit_test(altitude, velocity, time_f, dt)

        save_data = input("Save data? y/n > ").lower() == "y"
        if save_data:
            name = input("Enter name > ")
            file = OUT_PATH / (name + ".json")
            if file.exists():
                delete = input("Delete existing file? y/n > ").lower() == "y"
                if delete:
                    file.unlink()
                else:
                    file = OUT_PATH / (name + str(randint(1000000, 999999)) + ".json")

            with file.open("w") as f:
                saving = {"name": name}
                data_to_save = []
                for time, state in data:
                    data_to_save.append([time, state.to_json()])
                saving["data"] = data_to_save
                saving["world"] = world.to_json()
                saving["ship"] = ship.to_json()

                json.dump(saving, f)

        save = input("Save settings? y/n > ").lower() == "y"
        if save:
            option_data = options_data.get("testorbit")
            name = input("Enter name > ")

            if not option_data:
                options_data["testorbit"] = []

            options_data["testorbit"].append(
                {
                    "altitude": altitude,
                    "velocity": velocity,
                    "time_f": time_f,
                    "dt": dt,
                    "name": name,
                }
            )
            print("Saved!")

            with open(OPTIONS_PATH, "w") as f:
                json.dump(options_data, f)

    case 2:
        load = input("Enter filename to load no .json > ") + ".json"
        with open(OUT_PATH / load, "r") as f:
            data = json.load(f)

        import matplotlib.pyplot as plt

        times = [x[0] for x in data["data"]]
        states = [State.from_json(x[1]) for x in data["data"]]

        world = WorldConfig.from_json(data["world"])
        ship = ShipConfig.from_json(data["ship"])

        altitudes = [state.altitude(world) for state in states]
        velocities = [norm(state.vel) for state in states]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax1: Axes = axes[0]
        ax2: Axes = axes[1]

        ax1.plot(times, altitudes)
        ax1.set_title("Altitudes")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Altitude (m)")
        ax1.ticklabel_format(style="plain")

        ax2.plot(times, velocities)
        ax2.set_title("Velocities")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.ticklabel_format(style="plain")

        plt.tight_layout()
        plt.show()

        plotting.show_3d_plotly_trajectory(list(zip(times, states)), world, ship)
