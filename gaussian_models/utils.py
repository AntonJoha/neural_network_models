from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

EPS = 1e-6


def reparameterized_sample(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return mean + torch.randn_like(std) * std


def diag_kl(
    mean_post: torch.Tensor,
    std_post: torch.Tensor,
    mean_prior: torch.Tensor,
    std_prior: torch.Tensor,
) -> torch.Tensor:
    post_var = std_post.pow(2) + EPS
    prior_var = std_prior.pow(2) + EPS
    kld = (
        torch.log(prior_var)
        - torch.log(post_var)
        + (post_var + (mean_post - mean_prior).pow(2)) / prior_var
        - 1.0
    )
    return 0.5 * torch.sum(kld)


def gaussian_log_prob(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    var = std.pow(2) + EPS
    return -0.5 * (torch.log(2 * torch.pi * var) + (z - mean).pow(2) / var).sum(dim=-1)


def tdlgm_kl_term(
    mean: Sequence[torch.Tensor],
    r_factors: Sequence[torch.Tensor],
    *,
    use_stable_logdet: bool,
) -> torch.Tensor:
    matrix_size = mean[0].size(0) * mean[0].size(1)
    kl = torch.zeros((), device=mean[0].device, dtype=mean[0].dtype)

    eye = None
    if use_stable_logdet:
        eye = torch.eye(
            r_factors[0].size(-1),
            device=r_factors[0].device,
            dtype=r_factors[0].dtype,
        ).expand_as(r_factors[0])
    epsilon = max(EPS, torch.finfo(r_factors[0].dtype).eps)

    for m, r in zip(mean, r_factors, strict=True):
        c = r @ r.transpose(-2, -1)
        if use_stable_logdet:
            _, logdet = torch.linalg.slogdet(c + epsilon * eye)
        else:
            logdet = c.det().clamp(min=epsilon).log()

        kl = kl + (
            0.5
            * torch.sum(
                m.pow(2).sum(-1)
                + c.diagonal(dim1=-2, dim2=-1).sum(-1)
                - logdet
                - 1
            )
            / matrix_size
        )
    return kl


def tdlgm_state_regularization(
    states: Sequence[tuple[torch.Tensor, torch.Tensor]],
    next_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
    mse: nn.MSELoss,
    reg: float,
) -> torch.Tensor:
    amount = len(states) * len(states[0])
    state_reg = torch.zeros((), device=states[0][0].device, dtype=states[0][0].dtype)
    for a, b in zip(states, next_states, strict=True):
        state_reg = state_reg + reg * (mse(a[0], b[0]) + mse(a[1], b[1])) / amount
    return state_reg
