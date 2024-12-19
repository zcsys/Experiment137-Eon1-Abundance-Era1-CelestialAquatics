import torch
import pygame
import random
import math
import json
from base_vars import *
from helpers import *
from nn import *
from simulation import draw_dashed_circle
from diffusion import Grid

class Things:
    def __init__(self, thing_types = None, state_file = None):
        # Initialize font
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 12)

        if state_file:
            self.load_state(state_file)
            return

        # Main attributes
        self.thing_types = thing_types
        self.sizes = torch.tensor([THING_TYPES[x]["size"] for x in thing_types])
        self.positions = add_positions(len(thing_types))

        # Initialize tensor masks
        self.monad_mask = torch.tensor(
            [thing_type == "monad" for thing_type in self.thing_types]
        )
        self.energy_mask = torch.tensor(
            [thing_type == "energyUnit" for thing_type in self.thing_types]
        )
        self.structure_mask = torch.tensor(
            [thing_type == "structuralUnit" for thing_type in self.thing_types]
        )

        # Initialize state vars
        self.N = len(self.thing_types)
        self.Pop = self.monad_mask.sum().item()
        self.energies = torch.tensor(
            [THING_TYPES["monad"]["initial_energy"]
            for _ in range(self.Pop)]
        )
        self.E = self.energies.sum().item() // 1000
        self.colors = [THING_TYPES[x]["color"] for x in self.thing_types]
        self.memory = torch.zeros((self.Pop, 6), dtype = torch.float32)
        self.str_manipulations = torch.zeros((0, 2), dtype = torch.float32)

        # Initialize genomes and lineages
        # self.genomes = create_initial_genomes(self.Pop, 38, 26)
        genome_size = 6 + get_num_parameters_for_nn13(38, 26)
        print("Number of neurons:", 38 * 10 + 26)
        print("Genome size:", genome_size)
        self.genomes = torch.zeros((self.Pop, genome_size),
                                   dtype = torch.float32)
        self.lineages = [[0] for _ in range(self.Pop)]
        self.apply_genomes()

    def from_general_to_monad_idx(self, i):
        return self.monad_mask[:i].sum().item()

    def from_monad_to_general_idx(self, i):
        return torch.nonzero(self.monad_mask)[i].item()

    def get_generation(self, i):
        return self.lineages[i][0] + len(self.lineages[i])

    def apply_genomes(self):
        """Monad9x406 neurogenetics"""
        self.elemental_biases = torch.tanh(self.genomes[:, :6])
        self.nn = nn13(self.genomes[:, 6:], 38, 26)

    def mutate(self, i, probability = 0.1, strength = 1.):
        original_genome = self.genomes[i].clone()
        mutation_mask = torch.rand_like(original_genome) < probability
        mutations = torch.rand_like(original_genome) * 2 - 1
        return original_genome + mutation_mask * mutations * strength

    def sensory_inputs(self, grid):
        # Get the vicinity matrix
        indices, self.distances, self.diffs = vicinity(self.positions)

        # Gradient sensors
        y_pos = (self.positions[self.monad_mask][:, 1] // grid.cell_size).int()
        x_pos = (self.positions[self.monad_mask][:, 0] // grid.cell_size).int()
        grad_x, grad_y = grid.gradient()
        col1 = torch.zeros((self.Pop, 9), dtype = torch.float32)
        for channel in range(3):
            col1[:, channel * 3] = grid.grid[0, channel, y_pos, x_pos]
            col1[:, channel * 3 + 1] = grad_x[0, channel, y_pos, x_pos]
            col1[:, channel * 3 + 2] = grad_y[0, channel, y_pos, x_pos]
        col1 /= 255

        # Monads can interact with at most 8 structural units
        if self.monad_mask.any() and self.structure_mask.any():
            dist = self.distances[self.monad_mask][:, self.structure_mask]
            self.dist_mnd_str, self.structure_indices = torch.topk(
                dist.masked_fill(dist == 0, float('inf')),
                k = min(8, self.structure_mask.sum()),
                dim = 1,
                largest = False
            )

            col2  = (
                torch.gather(
                    self.diffs[self.monad_mask],
                    1,
                    self.structure_indices.unsqueeze(2).expand(-1, -1, 2)
                ) / (
                    torch.gather(
                        self.distances[self.monad_mask],
                        1,
                        self.structure_indices
                    ) ** 2 + epsilon
                ).unsqueeze(2)
            ).view(self.Pop, 16) * 8.
        else:
            col2 = torch.zeros((self.Pop, 16), dtype = torch.float32)

        # Combine the inputs to create the final input tensor
        self.input_vectors = torch.cat(
            (
                self.elemental_biases,
                col1,
                col2,
                (self.energies / 10000).unsqueeze(1),
                self.memory
            ),
            dim = 1
        ).view(self.Pop, 38, 1)

    def neural_action(self):
        return self.nn.forward(self.input_vectors)

    def random_action(self):
        numberOf_energyUnits = self.energy_mask.sum().item()
        if SYSTEM_HEAT == 0:
            return torch.tensor([[0, 0] for _ in range(numberOf_energyUnits)],
                                dtype = torch.float32)
        values = (torch.tensor(list(range(SYSTEM_HEAT)), dtype = torch.float32)
                  - (SYSTEM_HEAT - 1) / 2)
        weights = torch.ones(SYSTEM_HEAT, dtype = torch.float32)
        indices = torch.multinomial(
            weights,
            numberOf_energyUnits * 2,
            replacement = True
        ).view(numberOf_energyUnits, 2)
        return values[indices]

    def re_action(self, grid, neural_action):
        # Helper variables
        numberOf_structuralUnits = self.structure_mask.sum()
        movement_tensor = torch.zeros((numberOf_structuralUnits, 2),
                                      dtype = torch.float32)
        expanded_indices = self.structure_indices.unsqueeze(2).expand(-1, -1, 2)

        # Initialize force field
        force_field = torch.zeros_like(
            grid.grid
        ).repeat(2, 1, 1, 1).squeeze(1)
        indices = (self.positions[self.structure_mask] // grid.cell_size).long()

        # Calculate resource manipulations
        manipulation_contributions = (
            (
                torch.gather(
                    self.diffs[self.monad_mask][:, self.structure_mask],
                    1,
                    expanded_indices
                ) / (
                    torch.gather(
                        self.distances[self.monad_mask][:, self.structure_mask],
                        1,
                        self.structure_indices
                    ) ** 2 + epsilon
                ).unsqueeze(2)
            ) * neural_action[:, 8:16].unsqueeze(2)
        ) * 8.

        self.str_manipulations.scatter_add_(
            0,
            expanded_indices.view(-1, 2),
            manipulation_contributions.view(-1, 2)
        ).clamp_(-10, 10)

        # Calculate and apply force field with diffusion
        for i in range(2): # For vertical and horizontal axes
            force_field[i, 1][ # Green channel
                indices[:, 1], indices[:, 0]
            ] += self.str_manipulations[:, i]

        grid.diffuse(force_field)

        # Calculate movements
        movement_contributions = (
            torch.gather(
                self.diffs[self.monad_mask][:, self.structure_mask],
                1,
                expanded_indices
            ) / (
                torch.gather(
                    self.distances[self.monad_mask][:, self.structure_mask],
                    1,
                    self.structure_indices
                ) ** 2 + epsilon
            ).unsqueeze(2)
        ) * neural_action[:, 0:8].unsqueeze(2) * 8. # This scaling is because
                                                    # the minimum possible
        # Reduce energies                           # distance between a monad
        self.energies -= (                          # and a structural unit is 8
            movement_contributions.norm(dim = 2)
        ).sum(dim = 1) * 0.125 # This scaling is because a monad can interact
                               # with 8 different structural units
        # Return movements
        return movement_tensor.scatter_add(
            0,
            expanded_indices.view(-1, 2),
            movement_contributions.view(-1, 2)
        )

    def trace(self, grid):
        # Initialize force field
        force_field = torch.zeros_like(
            grid.grid
        ).repeat(2, 1, 1, 1).squeeze(1)

        # Monads to leave blue traces
        indices = (self.positions[self.monad_mask] // grid.cell_size).long()
        for i in range(2):
            force_field[i, 2][
                indices[:, 1], indices[:, 0]
            ] += torch.rand((self.Pop,), dtype = torch.float32) * 20 - 10

        # Energy units to leave red traces
        indices = (self.positions[self.energy_mask] // grid.cell_size).long()
        for i in range(2):
            force_field[i, 0][
                indices[:, 1], indices[:, 0]
            ] += torch.rand((self.energy_mask.sum(),),
                            dtype = torch.float32) * 20 - 10

        # Apply the field
        grid.apply_forces(force_field)

    def gradient_move(self, neural_action):
        return (
            neural_action[:, 0].unsqueeze(1) *
            self.input_vectors[:, 7:9].squeeze(2) +
            neural_action[:, 1].unsqueeze(1) *
            self.input_vectors[:, 10:12].squeeze(2) +
            neural_action[:, 2].unsqueeze(1) *
            self.input_vectors[:, 13:14].squeeze(2)
        ) * 10.

    def final_action(self, grid):
        # Update sensory inputs
        self.sensory_inputs(grid)

        # Initialize the movement tensor for this step
        if self.N > 0:
            self.movement_tensor = torch.tensor([[0., 0.]
                                                 for _ in range(self.N)])

        # Monad movements & internal state
        if self.monad_mask.any():
            neural_action = self.neural_action()
            self.movement_tensor[self.monad_mask] = self.gradient_move(
                neural_action[:, :3]
            )
            self.memory = neural_action[:, 20:26]

        # Fetch energyUnit movements
        if self.energy_mask.any():
            self.movement_tensor[self.energy_mask] = self.random_action()

        # Fetch structuralUnit reactions
        if self.structure_mask.any():
            if self.Pop > 0:
                self.movement_tensor[self.structure_mask] = self.re_action(
                    grid,
                    neural_action[:, 4:20]
                )
            else:
                self.movement_tensor[self.structure_mask] = torch.zeros(
                    (self.structure_mask.sum(), 2),
                    dtype = torch.float32
                )

        # Monads and energy units to leave trace
        self.trace(grid)

        # Auto-fission
        if self.monad_mask.any():
            random_gen = torch.rand(self.Pop)
            to_divide = neural_action[:, 3] > random_gen
            for i in to_divide.nonzero():
                self.monad_division(i.item())

        # Apply movements
        self.update_positions()

        # Update total monad energy
        self.E = self.energies.sum().item() // 1000

    def update_positions(self):
        provisional_positions = self.positions + self.movement_tensor

        # Apply toroidal boundaries
        provisional_positions[:, 0] %= SIMUL_WIDTH
        provisional_positions[:, 0] = torch.where(
            provisional_positions[:, 0] >= SIMUL_WIDTH,
            torch.tensor(0.),
            provisional_positions[:, 0]
        )
        provisional_positions[:, 1] %= SIMUL_HEIGHT
        provisional_positions[:, 1] = torch.where(
            provisional_positions[:, 1] >= SIMUL_HEIGHT,
            torch.tensor(0.),
            provisional_positions[:, 1]
        )

        # Get neighboring things
        indices, distances, diffs = vicinity(provisional_positions)

        # Monad-monad collisions
        dist = distances[self.monad_mask][:, self.monad_mask]
        collision_mask = (
            (0. < dist) & (dist < THING_TYPES["monad"]["size"] * 2)
        ).any(dim = 1)

        # StructureUnit-anything collisions
        size_sums = (
            self.sizes + THING_TYPES["structuralUnit"]["size"]
        ).unsqueeze(1)

        dist = distances[:, self.structure_mask]
        collision_mask_str = (
            (0. < dist) &
            (dist < size_sums)
        ).any(dim = 1)

        dist = distances[self.structure_mask]
        collision_mask_str[self.structure_mask] = (
            (0. < dist) &
            (dist < size_sums[self.structure_mask])
        ).any(dim = 1)

        # Check energy levels
        movement_magnitudes = torch.norm(
            self.movement_tensor[self.monad_mask],
            dim = 1
        )
        enough_energy = self.energies >= movement_magnitudes

        # Construct final apply mask
        final_apply_mask = ~collision_mask_str
        final_apply_mask[self.monad_mask] = (
            ~collision_mask_str[self.monad_mask] &
            ~collision_mask &
            enough_energy
        )

        # Apply the movements
        self.positions = torch.where(
            final_apply_mask.unsqueeze(1),
            provisional_positions,
            self.positions
        )

        # Reduce energies from monads
        actual_magnitudes = torch.where(
            final_apply_mask[self.monad_mask],
            movement_magnitudes,
            torch.tensor([0.])
        )
        self.energies -= actual_magnitudes

        # EnergyUnit-monad collisions
        energy_monad_dist = distances[self.energy_mask][:, self.monad_mask]
        collision_mask = (
            (0. < energy_monad_dist) &
            (energy_monad_dist < (THING_TYPES["monad"]["size"] +
                                 THING_TYPES["energyUnit"]["size"]))
        )

        if collision_mask.any():
            energy_idx, monad_idx = collision_mask.nonzero(as_tuple = True)
            energy_per_monad = (
                UNIT_ENERGY / collision_mask[energy_idx].sum(dim = 1)
            )
            self.energies.scatter_add_(
                0,
                monad_idx,
                energy_per_monad
            )
            energy_idx_general = torch.where(self.energy_mask)[0][energy_idx]
            self.remove_energyUnits(unique(energy_idx_general.tolist()))

    def monad_division(self, i):
        # Set out main attributes and see if division is possible
        thing_type = "monad"
        initial_energy = self.energies[i] / 2
        if (initial_energy <
            torch.tensor(THING_TYPES[thing_type]["initial_energy"])):
            return 0
        size = THING_TYPES[thing_type]["size"]
        idx = self.from_monad_to_general_idx(i)
        x, y = tuple(self.positions[idx].tolist())
        angle = random.random() * 2 * math.pi
        new_position = torch.tensor([
            x + math.cos(angle) * (size + 1) * 2,
            y + math.sin(angle) * (size + 1) * 2
        ])
        dist_mnd = torch.norm(
            self.positions[self.monad_mask] - new_position, dim = 1
        )
        dist_str = torch.norm(
            self.positions[self.structure_mask] - new_position, dim = 1
        )
        if (new_position[0] < size or new_position[0] > SIMUL_WIDTH - size or
            new_position[1] < size or new_position[1] > SIMUL_HEIGHT - size or
            (dist_str < size + THING_TYPES["structuralUnit"]["size"]).any() or
            (dist_mnd < size * 2).any()):
            return 0

        # Create a new set of attributes
        self.thing_types.append(thing_type)
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(size).unsqueeze(0)
            ),
            dim = 0
        )
        self.positions = torch.cat(
            (
                self.positions,
                new_position.unsqueeze(0)
            ),
            dim = 0
        )
        self.energies[i] -= initial_energy
        self.energies = torch.cat(
            (
                self.energies,
                initial_energy.unsqueeze(0)
            ),
            dim = 0
        )
        self.memory = torch.cat(
            (
                self.memory,
                torch.zeros((1, 6), dtype = torch.float32)
            ),
            dim = 0
        )
        self.movement_tensor = torch.cat(
            (
                self.movement_tensor,
                torch.tensor([[0., 0.]])
            ),
            dim = 0
        )
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
                torch.tensor([True])
            ),
            dim = 0
        )
        self.energy_mask = torch.cat(
            (
                self.energy_mask,
                torch.tensor([False])
            ),
            dim = 0
        )
        self.structure_mask = torch.cat(
            (
                self.structure_mask,
                torch.tensor([False])
            ),
            dim = 0
        )
        self.N += 1
        self.Pop += 1

        # Mutate the old genome & apply the new genome
        idx = self.from_general_to_monad_idx(i)
        genome = self.mutate(idx)
        self.genomes = torch.cat(
            (
                self.genomes,
                genome.unsqueeze(0)
            ),
            dim = 0
        )
        self.apply_genomes()
        if genome is self.genomes[idx]:
            self.lineages.append(self.lineages[idx])
            self.colors.append(self.color[i])
        else:
            new_lineage = self.lineages[idx] + [0]
            while True:
                new_lineage[-1] += 1
                if new_lineage not in self.lineages:
                    break
            self.lineages.append(new_lineage)
            self.colors.append(get_color_by_genome(genome))
            # print(new_lineage)

        return 1

    def perish_monad(self, indices):
        for i in indices[::-1]:
            # Remove monad-only attributes
            self.genomes = remove_element(self.genomes, i)
            self.energies = remove_element(self.energies, i)
            self.memory = remove_element(self.memory, i)
            del self.lineages[i]

            # Get general index to remove universal attributes
            idx = self.from_monad_to_general_idx(i)

            # Update main attributes and state vars
            del self.thing_types[idx]
            del self.colors[idx]
            self.sizes = remove_element(self.sizes, idx)
            self.positions = remove_element(self.positions, idx)
            self.monad_mask = remove_element(self.monad_mask, idx)
            self.energy_mask = remove_element(self.energy_mask, idx)
            self.structure_mask = remove_element(self.structure_mask, idx)

        # Update collective state vars
        self.N -= len(indices)
        self.Pop -= len(indices)

        self.apply_genomes()

    def add_energyUnits(self, N):
        for _ in range(N):
            self.thing_types.append("energyUnit")
            self.colors.append(THING_TYPES["energyUnit"]["color"])
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(
                    [THING_TYPES["energyUnit"]["size"] for _ in range(N)]
                )
            ),
            dim = 0
        )
        self.positions = add_positions(N, self.positions)
        self.N += N
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.energy_mask = torch.cat(
            (
                self.energy_mask,
                torch.ones(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.structure_mask = torch.cat(
            (
                self.structure_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )

    def add_energyUnits_atGridCells(self, feature, threshold):
        cell_indices = (feature >= threshold).nonzero()
        occupied_grid_cells = self.positions // GRID_CELL_SIZE

        positions_to_add = torch.empty((0, 2), dtype = torch.float32)
        for y, x in cell_indices:
            new_pos = torch.tensor([x, y])
            if torch.any(
                torch.all(occupied_grid_cells.int() == new_pos, dim = 1)
            ):
                continue
            else:
                positions_to_add = torch.cat(
                    (
                        positions_to_add,
                        (GRID_CELL_SIZE * (new_pos + 0.5)).unsqueeze(0)
                    ),
                    dim = 0
                )
                feature[y, x] = RESOURCE_TARGET

        N = len(positions_to_add)
        if N == 0:
            return
        self.N += N

        for _ in range(N):
            self.thing_types.append("energyUnit")
            self.colors.append(THING_TYPES["energyUnit"]["color"])
        self.positions = torch.cat(
            (
                self.positions,
                positions_to_add
            ),
            dim = 0
        )
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(
                    THING_TYPES["energyUnit"]["size"]
                ).expand(N)
            ),
            dim = 0
        )
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.energy_mask = torch.cat(
            (
                self.energy_mask,
                torch.ones(N, dtype = torch.bool)
            ),
            dim = 0
        )
        self.structure_mask = torch.cat(
            (
                self.structure_mask,
                torch.zeros(N, dtype = torch.bool)
            ),
            dim = 0
        )

    def remove_energyUnits(self, indices):
        for i in indices[::-1]:
            del self.thing_types[i]
            del self.colors[i]

        mask = torch.ones(self.N, dtype = torch.bool)
        mask[indices] = False
        self.N = mask.sum().item()

        self.sizes = self.sizes[mask]
        self.positions = self.positions[mask]
        self.monad_mask = self.monad_mask[mask]
        self.energy_mask = self.energy_mask[mask]
        self.structure_mask = self.structure_mask[mask]

    def draw(self, screen, show_info = True, show_sight = False,
             show_forces = True, show_communication = True):
        for i, pos in enumerate(self.positions):
            thing_type = self.thing_types[i]
            thing_color = self.colors[i]
            size = self.sizes[i].item()
            idx = self.from_general_to_monad_idx(i)

            if thing_type == "energyUnit":
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)
            elif thing_type == "monad":
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)
            elif thing_type == "structuralUnit":
                pygame.draw.circle(screen, thing_color, (int(pos[0].item()),
                                   int(pos[1].item())), size)

            if show_info and thing_type == "monad":
                # Show energy
                energy_text = self.energies[idx].item()
                if energy_text < 1000:
                    energy_text = str(int(energy_text))
                elif energy_text < 10000:
                    energy_text = f"{int(energy_text / 100) / 10:.1f}k"
                else:
                    energy_text = f"{int(energy_text / 1000)}k"
                energy_text = self.font.render(energy_text, True, colors["RGB"])
                energy_rect = energy_text.get_rect(
                    center = (
                        int(pos[0].item()),
                        int(pos[1].item() - 2 * size)
                    )
                )
                screen.blit(energy_text, energy_rect)

            if show_sight and thing_type == "monad":
                draw_dashed_circle(screen, self.colors[i], (int(pos[0].item()),
                                   int(pos[1].item())), SIGHT)

            try:
                input_vector_1 = self.input_vectors[idx, 0:2].squeeze(1)
                input_vector_2 = self.input_vectors[idx, 2:4].squeeze(1)
                input_vector_3 = self.input_vectors[idx, 4:6].squeeze(1)
                movement_vec = self.movement_tensor[i]
            except:
                show_forces = False
            if show_forces and thing_type == "monad":
                input_vector_1 /= torch.norm(input_vector_1, dim = 0) + epsilon
                input_vector_2 /= torch.norm(input_vector_2, dim = 0) + epsilon
                input_vector_3 /= torch.norm(input_vector_3, dim = 0) + epsilon
                movement_vec /= torch.norm(movement_vec, dim = 0) + epsilon

                end_pos_1 = pos + 2 * input_vector_1 * self.sizes[i]
                end_pos_2 = pos + 2 * input_vector_2 * self.sizes[i]
                end_pos_3 = pos + 2 * input_vector_3 * self.sizes[i]
                end_pos_4 = pos - 2 * movement_vec * self.sizes[i]

                pygame.draw.line(screen, colors["R"], (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_1[0].item()),
                                 int(end_pos_1[1].item())), 1)
                pygame.draw.line(screen, colors["GB"], (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_2[0].item()),
                                 int(end_pos_2[1].item())), 1)
                pygame.draw.line(screen, colors["RB"], (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_3[0].item()),
                                 int(end_pos_2[1].item())), 1)
                pygame.draw.line(screen, colors["RGB"], (int(pos[0].item()),
                                 int(pos[1].item())), (int(end_pos_4[0].item()),
                                 int(end_pos_3[1].item())), 2)

    def get_state(self):
        return {
            'types': self.thing_types,
            'positions': self.positions.tolist(),
            'energies': self.energies.tolist(),
            'genomes': self.genomes.tolist(),
            'str_manipulations': self.str_manipulations.tolist(),
            'lineages': self.lineages,
            'colors': self.colors,
            'memory': self.memory.tolist()
        }

    def load_state(self, state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)["things_state"]

        self.thing_types = state['types']
        self.sizes = torch.tensor(
            [THING_TYPES[x]["size"] for x in self.thing_types]
        )
        self.positions = torch.tensor(state['positions'])
        self.energies = torch.tensor(state['energies'])
        self.N = len(self.positions)
        self.genomes = torch.tensor(state['genomes'])
        self.lineages = state['lineages']
        self.colors = state['colors']
        self.str_manipulations = torch.tensor(state['str_manipulations'])
        self.memory = torch.tensor(state['memory'])

        self.monad_mask = torch.tensor(
            [thing_type == "monad" for thing_type in self.thing_types]
        )
        self.energy_mask = torch.tensor(
            [thing_type == "energyUnit" for thing_type in self.thing_types]
        )
        self.structure_mask = torch.tensor(
            [thing_type == "structuralUnit" for thing_type in self.thing_types]
        )
        self.Pop = self.monad_mask.sum().item()
        self.E = self.energies.sum().item() // 1000

        self.apply_genomes()

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 12)

    def add_structuralUnits(self, POP_STR = 1):
        self.thing_types += ["structuralUnit" for _ in range(POP_STR)]
        self.sizes = torch.cat(
            (
                self.sizes,
                torch.tensor(
                    [THING_TYPES["structuralUnit"]["size"]
                     for _ in range(POP_STR)]
                )
            ),
            dim = 0
        )
        self.positions = add_positions(POP_STR, self.positions)
        self.colors += [THING_TYPES["structuralUnit"]["color"]
                        for _ in range(POP_STR)]
        self.N += POP_STR
        self.monad_mask = torch.cat(
            (
                self.monad_mask,
                torch.zeros(POP_STR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.energy_mask = torch.cat(
            (
                self.energy_mask,
                torch.zeros(POP_STR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.structure_mask = torch.cat(
            (
                self.structure_mask,
                torch.ones(POP_STR, dtype = torch.bool)
            ),
            dim = 0
        )
        self.str_manipulations = torch.cat(
            (
                self.str_manipulations,
                torch.rand((POP_STR, 2), dtype = torch.float32) * 20 - 10
            ),
            dim = 0
        )
