import math

import torch

from foundry.utils.rigid import rot_vec_mul


def centre(X_L, X_exists_L):
    """
    Centers the coordinates X_L by subtracting the mean of existing atoms.
    """
    X_L = X_L.clone()
    B, D, L, _ = X_L.shape
    X_exists_L = X_exists_L.expand(-1, D, -1)
    X_L[X_exists_L] = X_L[X_exists_L] - torch.mean(
        X_L[X_exists_L], dim=-2, keepdim=True
    )
    X_L[~X_exists_L] = 0.0
    return X_L


def get_random_augmentation(X_L, s_trans, return_rotation=False):
    """
    Inputs:
        X_L [D, L, 3]: Batched atom coordinates
        s_trans (float): standard deviation of a global translation to be applied for each
            element in the batch
        return_rotation (bool): also return the sampled rotations R [B, D, 3, 3]
    """
    B, D, L, _ = X_L.shape
    R = uniform_random_rotation((B, D)).to(X_L.device)
    noise = s_trans * torch.normal(mean=0, std=1, size=(B, D, 1, 3)).to(X_L.device)
    X_L = rot_vec_mul(R[:, :, None], X_L) + noise
    return (X_L, R) if return_rotation else X_L


def centre_random_augmentation(X_L, X_exists_L, s_trans, return_rotation=False):
    """
    Inputs:
        X_L [B, D, L, 3]: Batched atom coordinates
        X_exists_L [B, D, L]: Batched atom existence mask
        s_trans (float): standard deviation of a global translation to be applied for each
            element in the batch
        return_rotation (bool): also return the sampled rotations R [B, D, 3, 3]
    Returns:
        Augmented X_L [B, D, L, 3] (and R, if requested)
    """
    X_L = centre(X_L, X_exists_L)
    return get_random_augmentation(X_L, s_trans, return_rotation)


def uniform_random_rotation(size):
    if isinstance(size, int):
        size = (size,)

    # Sample random angles
    theta_x = torch.rand(size) * 2 * math.pi
    theta_y = torch.rand(size) * 2 * math.pi
    theta_z = torch.rand(size) * 2 * math.pi

    cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
    cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)
    cos_z, sin_z = torch.cos(theta_z), torch.sin(theta_z)

    zeros = torch.zeros_like(cos_x)
    ones = torch.ones_like(cos_x)

    # Stack rotation matrices
    rotation_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_x, -sin_x], dim=-1),
            torch.stack([zeros, sin_x, cos_x], dim=-1),
        ],
        dim=-2,
    )

    rotation_y = torch.stack(
        [
            torch.stack([cos_y, zeros, sin_y], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_y, zeros, cos_y], dim=-1),
        ],
        dim=-2,
    )

    rotation_z = torch.stack(
        [
            torch.stack([cos_z, -sin_z, zeros], dim=-1),
            torch.stack([sin_z, cos_z, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    return torch.matmul(rotation_z, torch.matmul(rotation_y, rotation_x))
