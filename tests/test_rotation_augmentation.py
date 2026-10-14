"""Unit tests for foundry.utils.rotation_augmentation (batched [B, D, L, 3] layout used by rf3 sampling)."""

import pytest
import torch

from foundry.utils.rotation_augmentation import (
    centre,
    centre_random_augmentation,
    get_random_augmentation,
    uniform_random_rotation,
)


def _pdist(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x)


def test_uniform_random_rotation_is_proper_rotation():
    torch.manual_seed(0)
    R = uniform_random_rotation((2, 4))
    assert R.shape == (2, 4, 3, 3)
    assert torch.allclose(
        R @ R.transpose(-1, -2), torch.eye(3).expand(2, 4, 3, 3), atol=1e-5
    )
    assert torch.allclose(torch.linalg.det(R), torch.ones(2, 4), atol=1e-5)


def test_centre_present_atoms_and_zero_absent():
    torch.manual_seed(0)
    X = torch.randn(1, 2, 12, 3)
    mask = torch.ones(1, 1, 12, dtype=torch.bool)
    mask[0, 0, 0] = False
    X_before = X.clone()
    out = centre(X, mask)
    present = mask.expand(-1, 2, -1)
    assert torch.allclose(out[present].mean(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.all(out[~present] == 0.0)
    assert torch.equal(X, X_before)


def test_get_random_augmentation_is_rigid():
    torch.manual_seed(0)
    X = torch.randn(2, 3, 12, 3)
    out = get_random_augmentation(X, s_trans=2.0)
    assert out.shape == X.shape
    assert torch.allclose(_pdist(X), _pdist(out), atol=1e-4)


def test_centre_random_augmentation_return_rotation():
    torch.manual_seed(0)
    X = torch.randn(1, 3, 12, 3)
    mask = torch.ones(1, 1, 12, dtype=torch.bool)
    out, R = centre_random_augmentation(X, mask, s_trans=0.0, return_rotation=True)
    assert out.shape == X.shape and R.shape == (1, 3, 3, 3)
    assert torch.allclose(
        torch.einsum("bdij,bdlj->bdli", R, centre(X, mask)), out, atol=1e-4
    )
    assert isinstance(centre_random_augmentation(X, mask, s_trans=0.0), torch.Tensor)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
