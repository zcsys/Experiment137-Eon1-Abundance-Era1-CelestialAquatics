from base_vars import *
from base_vars import (N_TARGET, SYSTEM_HEAT, METABOLIC_ACTIVITY_CONSTANT,
                       AUTO_FISSION_THRESHOLD)
import torch

def Rules(simul, n):
    global N_TARGET, SYSTEM_HEAT, METABOLIC_ACTIVITY_CONSTANT, \
           AUTO_FISSION_THRESHOLD

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

    # Population control
    if 1 in n:
        if simul.period > 0 or simul.epoch >= 41:
            pass
        elif simul.epoch >= 40:
            N_TARGET = 300
            update_system_heat(3)
            AUTO_FISSION_THRESHOLD = 20000
            if simul.epoch == 40 and simul.age == 0 and simul.step == 1:
                simul.things.add_structuralUnits(14)
        elif simul.epoch >= 35:
            N_TARGET = 320
            update_system_heat(5)
            AUTO_FISSION_THRESHOLD = 18000
            if simul.age % 14 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        elif simul.epoch >= 30:
            N_TARGET = 340
            update_system_heat(5)
            AUTO_FISSION_THRESHOLD = 16000
            if simul.age % 14 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        elif simul.epoch >= 25:
            N_TARGET = 360
            update_system_heat(7)
            AUTO_FISSION_THRESHOLD = 15000
            if simul.age % 14 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        elif simul.epoch >= 20:
            N_TARGET = 380
            update_system_heat(7)
            AUTO_FISSION_THRESHOLD = 14000
            if simul.age % 14 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        elif simul.epoch >= 15:
            N_TARGET = 400
            update_system_heat(9)
            AUTO_FISSION_THRESHOLD = 13000
            if simul.age % 14 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        elif simul.epoch >= 10:
            N_TARGET = 425
            update_system_heat(9)
            AUTO_FISSION_THRESHOLD = 12000
            if simul.age % 14 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        elif simul.epoch >= 5:
            N_TARGET = 450
            update_system_heat(11)
            AUTO_FISSION_THRESHOLD = 11000
            if simul.age % 14 == 0 and simul.step == 1:
                simul.things.add_structuralUnits()
        else:
            N_TARGET = 500
            update_system_heat(11)
            AUTO_FISSION_THRESHOLD = 10000
            if simul.epoch == 0 and simul.age == 0 and simul.step == 1:
                simul.things.add_structuralUnits(16)

        if simul.things.N < N_TARGET:
            simul.things.add_energyUnits(N_TARGET - simul.things.N)

        if simul.things.E <= 100:
            METABOLIC_ACTIVITY_CONSTANT = 0.1
        elif 100 < simul.things.E <= 200:
            METABOLIC_ACTIVITY_CONSTANT = 0.1 + 0.009 * (simul.things.E - 100)
        elif 200 < simul.things.E:
            METABOLIC_ACTIVITY_CONSTANT = 1. + 0.09 * (simul.things.E - 200)

    # Resource management
    if 2 in n:
        if simul.period > 0 or simul.epoch > 0:
            simul.things.add_energyUnits_atGridCells(simul.grid.grid[0][1], 128)
