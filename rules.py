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
        if simul.period > 0 or simul.epoch > 0:
            simul.things.energies -= METABOLIC_ACTIVITY_CONSTANT
        else:
            simul.things.energies[50:] -= METABOLIC_ACTIVITY_CONSTANT
        to_remove = torch.nonzero(simul.things.energies <= 0)
        if len(to_remove) > 0:
            simul.things.perish_monad(to_remove.squeeze(1).tolist())
        simul.things.E = simul.things.energies.sum().item() // 1000

    # Incubation
    if 1 in n:
        pass

    # Population control
    if 2 in n:
        if simul.things.E <= 500:
            METABOLIC_ACTIVITY_CONSTANT = 0.1
        elif 500 < simul.things.E <= 600:
            METABOLIC_ACTIVITY_CONSTANT = 0.1 + 0.009 * (simul.things.E - 500)
        elif 600 < simul.things.E:
            METABOLIC_ACTIVITY_CONSTANT = 1. + 0.09 * (simul.things.E - 600)

    # Resource management
    if 3 in n:
        if simul.things.N < N_TARGET:
            simul.things.add_energyUnits_atGridCells(simul.grid.grid[0][1],
                                                     ENERGY_THRESHOLD)

    # Autogenetic breeding
    if 4 in n:
        pass
