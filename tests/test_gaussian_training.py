import unittest

import torch
import torch.optim as optim

from gaussian_models.tDLGM import tDLGM


def clone_parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def parameters_changed(before, module):
    after = list(module.parameters())
    return any(not torch.allclose(b, a.detach()) for b, a in zip(before, after, strict=True))


class TestGaussianTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = tDLGM(
            input_dim=4,
            hidden_size=8,
            latent_dim=4,
            output_dim=4,
            layers=1,
            seq_len=3,
        )
        self.optimizer = optim.Adam(self.model.get_parameters(), lr=1e-1)
        self.x = torch.randn(16, 3, 4)
        self.y = torch.randn(16, 1, 4)
        self.x_1 = torch.cat((self.x, self.y), dim=1)[:, 1:, :]

    def test_tdlgm_parameters_change(self):
        before = clone_parameters(self.model)
        self.model.train_step(self.x, self.x_1, self.y, self.optimizer)
        self.assertTrue(parameters_changed(before, self.model))

    def test_tdlgm_loss_decreases(self):
        initial_loss = self.model.get_loss(self.x, self.x_1, self.y)
        for _ in range(150):
            self.model.train_step(self.x, self.x_1, self.y, self.optimizer)
        final_loss = self.model.get_loss(self.x, self.x_1, self.y)
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
