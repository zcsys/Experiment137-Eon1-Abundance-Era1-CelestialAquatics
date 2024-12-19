import torch
import torch.nn.functional as F

class nn2:
    def __init__(self, weights, input_size, output_size):
        self.num_monads = weights.shape[0]
        L1_size = input_size * 4
        L2_size = max(input_size, 2 * output_size)

        assert weights.shape[1] == (
            L1_size * (input_size + 1) +
            L2_size * (L1_size + 1) +
            output_size * (L2_size + 1)
        ), "Weight size mismatch"

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

class nn03:
    def __init__(self, weights, input_size, output_size):
        self.num_monads = weights.shape[0]
        L1_size = input_size * 4
        L2_size = input_size * 3
        L3_size = input_size * 2

        assert weights.shape[1] == (
            L1_size * (input_size + 1) +
            L2_size * (L1_size + 1) +
            L3_size * (L2_size + 1) +
            output_size * (L3_size + 1)
        ), "Weight size mismatch"

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

        self.W3 = weights[
            :, pos:pos + L3_size * L2_size
        ].view(self.num_monads, L3_size, L2_size)
        pos += L3_size * L2_size
        self.B3 = weights[:, pos:pos + L3_size].unsqueeze(2)
        pos += L3_size

        self.Wo = weights[
            :, pos:pos + output_size * L3_size
        ].view(self.num_monads, output_size, L3_size)
        pos += output_size * L3_size
        self.Bo = weights[:, pos:].unsqueeze(2)

    def forward(self, inputs):
        ff_1 = torch.relu(self.W1 @ inputs + self.B1)
        ff_2 = torch.relu(self.W2 @ ff_1 + self.B2)
        ff_3 = F.leaky_relu(self.W3 @ ff_2 + self.B3, negative_slope = 0.01)
        return torch.tanh(self.Wo @ ff_3 + self.Bo).squeeze(2)

def get_num_parameters_for_nn13(input_size, output_size):
    L1_size = input_size * 4
    L2_size = input_size * 3
    L3_size = input_size * 2
    return (
        L1_size * (input_size + 1) +
        L2_size * (input_size + L1_size + 1) +
        L3_size * (input_size + L1_size + L2_size + 1) +
        output_size * (input_size + L1_size + L2_size + L3_size + 1)
    )

class nn13:
    def __init__(self, weights, input_size, output_size):
        self.num_monads = weights.shape[0]
        L1_size = input_size * 4
        L2_size = input_size * 3
        L3_size = input_size * 2

        num_parameters = get_num_parameters_for_nn13(input_size, output_size)
        assert weights.shape[1] == num_parameters, "Weight size mismatch"

        pos = 0
        self.W1 = weights[
            :, :L1_size * input_size
        ].view(self.num_monads, L1_size, input_size)
        pos += L1_size * input_size
        self.B1 = weights[:, pos:pos + L1_size].unsqueeze(2)
        pos += L1_size

        self.W2 = weights[
            :, pos:pos + L2_size * (input_size + L1_size)
        ].view(self.num_monads, L2_size, input_size + L1_size)
        pos += L2_size * (input_size + L1_size)
        self.B2 = weights[:, pos:pos + L2_size].unsqueeze(2)
        pos += L2_size

        self.W3 = weights[
            :, pos:pos + L3_size * (input_size + L1_size + L2_size)
        ].view(self.num_monads, L3_size, input_size + L1_size + L2_size)
        pos += L3_size * (input_size + L1_size + L2_size)
        self.B3 = weights[:, pos:pos + L3_size].unsqueeze(2)
        pos += L3_size

        self.Wo = weights[
            :, pos:pos + output_size * (input_size + L1_size + L2_size +
                                        L3_size)
        ].view(self.num_monads, output_size, input_size + L1_size + L2_size +
                                             L3_size)
        pos += output_size * (input_size + L1_size + L2_size + L3_size)
        self.Bo = weights[:, pos:].unsqueeze(2)

    def forward(self, inputs):
        ff_1 = torch.relu(self.W1 @ inputs + self.B1)

        cat_2 = torch.cat([inputs, ff_1], dim = 1)
        ff_2 = torch.relu(self.W2 @ cat_2 + self.B2)

        cat_3 = torch.cat([inputs, ff_1, ff_2], dim = 1)
        ff_3 = F.leaky_relu(self.W3 @ cat_3 + self.B3, negative_slope = 0.01)

        cat_o = torch.cat([inputs, ff_1, ff_2, ff_3], dim = 1)
        return torch.tanh(self.Wo @ cat_o + self.Bo).squeeze(2)
