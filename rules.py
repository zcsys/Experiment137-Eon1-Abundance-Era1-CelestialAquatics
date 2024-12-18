from base_vars import *
from base_vars import METABOLIC_ACTIVITY_CONSTANT, AUTO_FISSION_THRESHOLD

import torch

def Rules(simul, n):
    global METABOLIC_ACTIVITY_CONSTANT, AUTO_FISSION_THRESHOLD

    # Coming into existence and perishing
    if 0 in n:
        fission_mask = simul.things.energies >= AUTO_FISSION_THRESHOLD
        for i, mask in enumerate(fission_mask):
            if mask:
                simul.things.monad_division(i)
        simul.things.energies -= METABOLIC_ACTIVITY_CONSTANT
        to_remove = torch.nonzero(simul.things.energies <= 0)
        if len(to_remove) > 0:
            simul.things.perish_monad(to_remove.squeeze(1).tolist())
        simul.things.E = simul.things.energies.sum().item() // 1000

    # Incubation
    if 1 in n:
        if simul.period > 0:
            pass
        elif simul.epoch >= 60:
            update_system_heat(3)
            update_energy_threshold(120)
            if simul.age % 40 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        elif simul.epoch >= 40:
            update_system_heat(5)
            update_energy_threshold(110)
            if simul.age % 40 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        elif simul.epoch >= 20:
            update_system_heat(7)
            update_energy_threshold(105)
            if simul.age % 40 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        else:
            update_system_heat(9)
            update_energy_threshold(100)
            if simul.epoch == 0 and simul.age == 0 and simul.step == 1:
                simul.things.add_structuralUnits(120)

    # Population control
    if 2 in n:
        if simul.things.E <= 100:
            METABOLIC_ACTIVITY_CONSTANT = 0.1
        elif 100 < simul.things.E <= 200:
            METABOLIC_ACTIVITY_CONSTANT = 0.1 + 0.009 * (simul.things.E - 100)
        elif 200 < simul.things.E:
            METABOLIC_ACTIVITY_CONSTANT = 1. + 0.09 * (simul.things.E - 200)

    # Resource management
    if 3 in n:
        simul.things.add_energyUnits_atGridCells(simul.grid.grid[0][1],
                                                 ENERGY_THRESHOLD)
