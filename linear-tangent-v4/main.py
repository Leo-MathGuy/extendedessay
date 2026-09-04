import json
import logging
from random import randint

import logs

l = logging.getLogger("l")

import dynamics_examples
from pathlib import Path

print("Select option")
print(" # 1 - Simulate altitude + velocity orbit")
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

        data = dynamics_examples.test_orbit(altitude, velocity, time_f, dt)

        save_data = input("Save data? y/n > ").lower() == "y"
        if save_data:
            name = input("Enter name > ")
            file = OUT_PATH / name
            if file.exists():
                delete = input("Delete existing file? y/n > ").lower() == "y"
                if delete:
                    file.unlink()
                else:
                    file = OUT_PATH / (name + str(randint(1000000, 999999)))

            with file.open("w") as f:
                data = {"name": name, "data": data}
                json.dump(data, f)

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
