import unittest

import torch
import torch.optim as optim

from gaussian_models.vae_variants import (
    IWAEVRNN,
    NFVRNN,
    SRNN,
    BetaDLGM,
    BetatDLGM,
    ConditionalVRNN,
    KalmanVAE,
    LadderVAE,
)

TEST_SEED = 7


def clone_parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def parameters_changed(before, module):
    after = list(module.parameters())
    return any(
        not torch.allclose(b, a.detach()) for b, a in zip(before, after, strict=True)
    )


class TestVAEVariantsTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(TEST_SEED)
        self.x = torch.randn(12, 3, 4)
        self.y = torch.randn(12, 1, 4)
        self.x_1 = torch.cat((self.x, self.y), dim=1)[:, 1:, :]
        self.cond = torch.randn(12, 2)

    def _assert_train_step_updates(self, model, *, condition=None):
        optimizer = optim.Adam(model.get_parameters(), lr=5e-2)
        before = clone_parameters(model)
        if condition is None:
            loss = model.train_step(self.x, self.x_1, self.y, optimizer)
            pred = model.forward(self.x)
        else:
            loss = model.train_step(self.x, self.x_1, self.y, optimizer, condition=condition)
            pred = model.forward(self.x, condition=condition)
        self.assertIsInstance(loss, float)
        self.assertTrue(parameters_changed(before, model))
        self.assertEqual(pred.size(0), self.y.size(0))
        self.assertEqual(pred.size(-1), self.y.size(-1))
        self.assertIn(pred.dim(), {2, 3})

    def test_beta_dlgm_train_step(self):
        model = BetaDLGM(input_dim=4, hidden_size=8, latent_dim=4, output_dim=4, layers=1, seq_len=3, beta=3.0)
        self._assert_train_step_updates(model)

    def test_beta_tdlgm_train_step(self):
        model = BetatDLGM(input_dim=4, hidden_size=8, latent_dim=4, output_dim=4, layers=1, seq_len=3, beta=2.5)
        self._assert_train_step_updates(model)

    def test_conditional_vrnn_train_step(self):
        model = ConditionalVRNN(
            input_dim=4,
            hidden_size=8,
            latent_dim=4,
            output_dim=4,
            layers=1,
            seq_len=3,
            cond_dim=2,
        )
        self._assert_train_step_updates(model, condition=self.cond)

    def test_iwae_vrnn_train_step(self):
        model = IWAEVRNN(
            input_dim=4,
            hidden_size=8,
            latent_dim=4,
            output_dim=4,
            layers=1,
            seq_len=3,
            importance_samples=3,
        )
        self._assert_train_step_updates(model)

    def test_ladder_vae_train_step(self):
        model = LadderVAE(input_dim=4, hidden_size=8, latent_dim=4, output_dim=4, layers=1, seq_len=3)
        self._assert_train_step_updates(model)

    def test_kalman_vae_train_step(self):
        model = KalmanVAE(input_dim=4, hidden_size=8, latent_dim=4, output_dim=4, layers=1, seq_len=3)
        self._assert_train_step_updates(model)

    def test_srnn_train_step(self):
        model = SRNN(input_dim=4, hidden_size=8, latent_dim=4, output_dim=4, layers=1, seq_len=3)
        self._assert_train_step_updates(model)

    def test_nf_vrnn_train_step(self):
        model = NFVRNN(input_dim=4, hidden_size=8, latent_dim=4, output_dim=4, layers=1, seq_len=3, flow_layers=2)
        self._assert_train_step_updates(model)


if __name__ == "__main__":
    unittest.main()
