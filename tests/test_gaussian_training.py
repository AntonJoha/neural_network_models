import unittest
import math

import torch
import torch.optim as optim

from gaussian_models.DLGM import DLGM
from gaussian_models.VRNN import VRNN
from gaussian_models.tDLGM import tDLGM
from gaussian_models.tDLGM_attention import tDLGMAttention


def clone_parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def parameters_changed(before, module):
    after = list(module.parameters())
    return any(
        not torch.allclose(b, a.detach()) for b, a in zip(before, after, strict=True)
    )


class TestDLGMTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = DLGM(input_dim=4, hidden_size=8, latent_dim=4, output_dim=4, layers=1, seq_len=3)
        self.optimizer = optim.Adam(self.model.get_parameters(), lr=1e-1)
        self.x = torch.randn(16, 3, 4)
        self.y = torch.randn(16, 1, 4)
        # x_1 is the one-step-shifted sequence used by recognition/training.
        self.x_1 = torch.cat((self.x, self.y), dim=1)[:, 1:, :]

    def test_dlmg_parameters_change(self):
        before = clone_parameters(self.model)
        self.model.train_step(self.x, self.x_1, self.y, self.optimizer)
        self.assertTrue(parameters_changed(before, self.model))

    def test_dlmg_loss_decreases(self):
        initial_loss = self.model.get_loss(self.x, self.x_1, self.y)
        for _ in range(150):
            self.model.train_step(self.x, self.x_1, self.y, self.optimizer)
        final_loss = self.model.get_loss(self.x, self.x_1, self.y)
        self.assertLess(final_loss, initial_loss)

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


class TestGaussianAttentionTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = tDLGMAttention(
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

    def test_tdlgm_attention_parameters_change(self):
        before = clone_parameters(self.model)
        self.model.train_step(self.x, self.x_1, self.y, self.optimizer)
        self.assertTrue(parameters_changed(before, self.model))

    def test_tdlgm_attention_loss_decreases(self):
        initial_loss = self.model.get_loss(self.x, self.x_1, self.y)
        for _ in range(150):
            self.model.train_step(self.x, self.x_1, self.y, self.optimizer)
        final_loss = self.model.get_loss(self.x, self.x_1, self.y)
        self.assertLess(final_loss, initial_loss)


class TestVRNNTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = VRNN(
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

    def test_vrnn_parameters_change(self):
        before = clone_parameters(self.model)
        self.model.train_step(self.x, self.x_1, self.y, self.optimizer)
        self.assertTrue(parameters_changed(before, self.model))

    def test_vrnn_loss_decreases(self):
        initial_loss = self.model.get_loss(self.x, self.x_1, self.y)
        for _ in range(150):
            self.model.train_step(self.x, self.x_1, self.y, self.optimizer)
        final_loss = self.model.get_loss(self.x, self.x_1, self.y)
        self.assertLess(final_loss, initial_loss)

    def test_vrnn_input_shape_validation(self):
        bad_x_1 = torch.randn(16, 3, 5)
        with self.assertRaises(ValueError):
            self.model.get_loss(self.x, bad_x_1, self.y)

    def test_vrnn_target_shape_validation(self):
        bad_y = torch.randn(16, 1, 5)
        with self.assertRaises(ValueError):
            self.model.get_loss(self.x, self.x_1, bad_y)

    def test_vrnn_loss_is_finite_for_zero_inputs(self):
        zero_x = torch.zeros_like(self.x)
        zero_y = torch.zeros_like(self.y)
        zero_x_1 = torch.cat((zero_x, zero_y), dim=1)[:, 1:, :]
        loss = self.model.get_loss(zero_x, zero_x_1, zero_y)
        self.assertIsInstance(loss, float)
        self.assertTrue(math.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
