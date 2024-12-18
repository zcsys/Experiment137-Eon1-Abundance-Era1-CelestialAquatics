import torch

class nn2:
    def __init__(self, weights, input_size, output_size):
        # Get the number of monads
        self.num_monads = weights.shape[0]

        # Set up weights
        L1_size = input_size * 4
        L2_size = max(input_size, 2 * output_size)

        # Size check
        # 4*34*(34+1)+34*(4*34+1)+23*(2*34+1) = 11005
        assert (
            weights.shape[1] ==
            (
                L1_size * (input_size + 1) +
                L2_size * (L1_size + 1) +
                output_size * (L2_size + 1)
            ),
            "Weight size mismatch"
        )

        pos = 0
        self.W1 = weights[
            :, :L1_size * input_size
        ].view(self.num_monads, L1_size, input_size)
        pos += L1_size * input_size
        self.B1 = weights[:, pos:pos + L1_size].unsqueeze(2)
        pos += L1_size

        self.W2 = weights[
            :, pos:pos + L2_size * L1_size
        ].view(self.num_monads, L2_size, L1_size)
        pos += L2_size * L1_size
        self.B2 = weights[:, pos:pos + L2_size].unsqueeze(2)
        pos += L2_size

        self.Wo = weights[
            :, pos:pos + output_size * L2_size
        ].view(self.num_monads, output_size, L2_size)
        pos += output_size * L2_size
        self.Bo = weights[:, pos:].unsqueeze(2)

    def forward(self, inputs):
        ff_1 = torch.relu(self.W1 @ inputs + self.B1)
        ff_2 = torch.relu(self.W2 @ ff_1 + self.B2)
        return torch.tanh(self.Wo @ ff_2 + self.Bo).squeeze(2)
