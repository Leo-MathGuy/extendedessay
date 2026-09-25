import json
import logging
from random import randint

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from scipy.linalg import norm

import constants
import lintangoptimize
import plotting
from data_types import State, ShipConfig, WorldConfig

# noinspection PyUnusedImports
import logs
from plotting import show_penalty_mesh

l = logging.getLogger("l")

import dynamics_examples
from pathlib import Path

print("Select option")
print(" # 1 - Simulate altitude + velocity orbit")
print(" # 2 - Analyze 3d graph")
print(" # 3 - Linear optimize descent manual")
print(" # 4 - 3d graph for lintan")
print(" # 5 - Lintan a/b")
print(" # 6 - a/b lintan derivatives")
print(" # 7 - a/b lintan matrix")
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

    case 3:
        load = input("Load saved settings? y/n > ").lower() == "y"

        dt = None
        a = None
        b = None
        t = None

        if load:
            option_data = options_data.get("manuallintan")
            if option_data is None:
                print("Failed to load settings")
                load = False
            else:
                for i, option in enumerate(option_data):
                    print(f" - Option {i} ({option.get('name')}):")
                    print(f" - - dt {option.get('dt')}s")
                    print(f" - - a {round(option.get('a'),7)}")
                    print(f" - - b {round(option.get('b'),7)}")
                    print(f" - - t {round(option.get('t'),7)}")

                selected = int(input("Select option > "))
                data = option_data[selected]
                dt = data["dt"]
                a = data["a"]
                b = data["b"]
                t = data["t"]

        if not load:
            a = float(input("Enter a > "))
            b = float(input("Enter b > "))
            t = float(input("Enter t > "))
            dt = float(input("Enter dt > "))
            print()

        data, world, ship = lintangoptimize.lintang_orbit_test(dt, a, b, t, log=2)

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
            option_data = options_data.get("manuallintan")
            name = input("Enter name > ")

            if not option_data:
                options_data["manuallintan"] = []

            options_data["manuallintan"].append(
                {
                    "a": a,
                    "b": b,
                    "t": t,
                    "dt": dt,
                    "name": name,
                }
            )
            print("Saved!")

            with open(OPTIONS_PATH, "w") as f:
                json.dump(options_data, f)

    case 4:
        load = input("Enter filename to load no .json > ") + ".json"
        with open(OUT_PATH / load, "r") as f:
            data = json.load(f)

        times = [x[0] for x in data["data"]]
        states = [State.from_json(x[1]) for x in data["data"]]

        world = WorldConfig.from_json(data["world"])
        ship = ShipConfig.from_json(data["ship"])

        altitudes = [state.altitude(world) for state in states]
        velocities = [norm(state.vel) for state in states]

        masses = [state.mass for state in states]
        angles = [state.extra_data["angle"] for state in states]

        fig, axes = plt.subplots(2, 2, figsize=(10, 5))
        ax1: Axes = axes[0, 0]
        ax2: Axes = axes[1, 0]
        ax3: Axes = axes[0, 1]
        ax4: Axes = axes[1, 1]

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

        ax3.plot(times, masses)
        ax3.set_title("Mass vs Time")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Mass (kg)")
        ax3.ticklabel_format(style="plain")

        ax4.plot(times[1:], angles[1:])
        ax4.set_title("Pitch angles")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Angle (degrees)")
        ax4.ticklabel_format(style="plain")

        plt.tight_layout()
        plt.show()

        plotting.show_3d_plotly_trajectory_lintan(list(zip(times, states)), world, ship)

    case 5:
        load = input("Load saved settings? y/n > ").lower() == "y"

        dt = None
        a = None
        b = None
        t = None

        if load:
            option_data = options_data.get("manuallintanab")
            if option_data is None:
                print("Failed to load settings")
                load = False
            else:
                for i, option in enumerate(option_data):
                    print(f" - Option {i} ({option.get('name')}):")
                    print(f" - - dt {option.get('dt')}s")
                    print(f" - - a {round(option.get('a'),7)}")
                    print(f" - - b {round(option.get('b'),7)}")
                    print(f" - - t {round(option.get('t'),7)}")

                selected = int(input("Select option > "))
                data = option_data[selected]
                dt = data["dt"]
                a = data["a"]
                b = data["b"]
                t = data["t"]

        if not load:
            a = float(input("Enter a > "))
            b = float(input("Enter b > "))
            t = float(input("Enter initial t > "))
            dt = float(input("Enter dt > "))
            print()

        res = lintangoptimize.do_a_b_pair(dt, a, b, init_time=t, log=1)
        if res is None:
            l.error("Failed to find start time")
            exit()

        data, world, ship = res

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
            option_data = options_data.get("manuallintanab")
            name = input("Enter name > ")

            if not option_data:
                options_data["manuallintanab"] = []

            options_data["manuallintanab"].append(
                {
                    "a": a,
                    "b": b,
                    "t": t,
                    "dt": dt,
                    "name": name,
                }
            )
            print("Saved!")

            with open(OPTIONS_PATH, "w") as f:
                json.dump(options_data, f)
    case 6:
        a = float(input("Enter a > "))
        b = float(input("Enter b > "))
        t_i = float(input("Enter t_i > "))
        t_f = float(input("Enter t_f > "))
        t_s = int(input("Enter steps > "))

        times = np.linspace(t_i, t_f, t_s)
        derivatives = np.ndarray((len(times),), dtype=float)
        values = np.ndarray((len(times),), dtype=float)

        print(len(times))
        for i, time in enumerate(times):
            derivatives[i], le, ri = lintangoptimize.find_derivative(
                constants.INITIAL_ALTITUDE,
                constants.INITIAL_SPEED,
                1800,
                a,
                b,
                time,
                1,
                0.2,
                0,
            )
            values[i] = (le + ri) / 2
            l.info(f"Finished {i}/{round(t_s)}")
        l.info("Done")

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1: Axes
        ax2: Axes

        ax1.plot(times, derivatives)

        ax1.set_xlabel("Time")
        ax1.set_ylabel("df/dx")

        ax2.plot(times, values)

        ax2.set_xlabel("Time")
        ax2.set_ylabel("f(x)")

        plt.tight_layout()
        plt.show()
    case 7:
        a_i = float(input("Enter a_i > "))
        a_f = float(input("Enter a_f > "))
        a_s = int(input("Enter a_steps > "))
        b_i = float(input("Enter b_i > "))
        b_f = float(input("Enter b_f > "))
        b_s = int(input("Enter b_steps > "))
        dt = float(input("Enter dt > "))

        a_list = np.linspace(a_i, a_f, a_s)
        b_list = np.linspace(b_i, b_f, b_s)

        A, B = np.meshgrid(a_list, b_list, indexing="ij")
        matrix = np.stack([A, B], axis=-1)

        result = np.ndarray((a_s, b_s), dtype=float)

        print(matrix)

        for i, a in enumerate(matrix):
            for j, b in enumerate(a):
                res = lintangoptimize.do_a_b_pair(dt, b[0], b[1], init_time=900, log=0)
                if res is not None:
                    penalty = res[0][-1][1].extra_data.get("penalty", -1)
                else:
                    penalty = -1
                l.info(f"Finished {i * b_s + j + 1} / {a_s * b_s} - " + str(penalty))

                result[i, j] = penalty

        l.info("Succeded")

        show_penalty_mesh(a_list, b_list, result)
