import sys

SIMUL_WIDTH = 1920
SIMUL_HEIGHT = 1080
MENU_WIDTH = 180
SCREEN_WIDTH = SIMUL_WIDTH + MENU_WIDTH
SCREEN_HEIGHT = SIMUL_HEIGHT

SIGHT = 60
N_TARGET = 300
AUTO_FISSION_THRESHOLD = 20000
METABOLIC_ACTIVITY_CONSTANT = 0.1
UNIT_ENERGY = 1000
SYSTEM_HEAT = 3
GRID_CELL_SIZE = 10
RESOURCE_TARGET = 5e+6

colors = {
    "0": (0, 0, 0),
    "b": (0, 0, 127),
    "B": (0, 0, 254),
    "g": (0, 127, 0),
    "gb": (0, 127, 127),
    "gB": (0, 127, 254),
    "G": (0, 254, 0),
    "Gb": (0, 254, 127),
    "GB": (0, 254, 254),
    "r": (127, 0 ,0),
    "rb": (127, 0, 127),
    "rB": (127, 0, 254),
    "rg": (127, 127, 0),
    "rgb": (127, 127, 127),
    "rgB": (127, 127, 254),
    "rG": (127, 254, 0),
    "rGb": (127, 254, 127),
    "rGB": (127, 254, 254),
    "R": (254, 0, 0),
    "Rb": (254, 0, 127),
    "RB": (254, 0, 254),
    "Rg": (254, 127, 0),
    "Rgb": (254, 127, 127),
    "RgB": (254, 127, 254),
    "RG": (254, 254, 0),
    "RGb": (254, 254, 127),
    "RGB": (254, 254, 254),
}

THING_TYPES = {
    "monad": {
        "color": colors["rgb"],
        "size": 5,
        "initial_energy": 1000.
    },
    "energyUnit": {
        "color": colors["G"],
        "size": 1
    },
    "structuralUnit": {
        "color": colors["R"],
        "size": 3
    },
    "memoryUnit": {
        "color": colors["B"],
        "size": 10
    }
}

def update_system_heat(new_value):
    global SYSTEM_HEAT
    SYSTEM_HEAT = new_value
    for module in sys.modules.values():
        if hasattr(module, "SYSTEM_HEAT"):
            module.SYSTEM_HEAT = new_value