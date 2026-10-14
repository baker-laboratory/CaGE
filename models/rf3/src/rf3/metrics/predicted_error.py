from typing import Any

import numpy as np
import torch
from rf3.chemical import NFRAMES, frame_indices
from rf3.metrics.metric_utils import (
    find_bin_midpoints,
    unbin_logits,
)
from rf3.utils.frames import (
    get_frames,
    mask_unresolved_frames_batched,
    rigid_from_3_points,
)

from foundry.metrics.metric import Metric


def cross_chain_pair_mask(
    chain_ids: torch.Tensor | np.ndarray | list,
    valid_pairs: torch.Tensor,
) -> torch.Tensor:
    """Boolean (I, I) mask: valid token pairs on different chains.

    ``chain_ids`` may arrive as a length-I vector, or with a leading batch dim of 1
    (dataloader wrap of string arrays). That extra dim must be squeezed — otherwise
    ``ids[None] != ids[:, None]`` becomes an elementwise self-compare and is all False.
    """
    n_tokens = valid_pairs.shape[0]
    if isinstance(chain_ids, torch.Tensor):
        chain = chain_ids.detach()
        while chain.ndim > 1 and chain.shape[0] == 1:
            chain = chain[0]
        chain = chain.reshape(-1).to(device=valid_pairs.device)
        if chain.numel() != n_tokens:
            return torch.zeros_like(valid_pairs)
        cross = chain[None, :] != chain[:, None]
    else:
        chain_np = np.asarray(chain_ids)
        while chain_np.ndim > 1 and chain_np.shape[0] == 1:
            chain_np = chain_np[0]
        chain_np = np.asarray(chain_np).reshape(-1)
        if chain_np.shape[0] != n_tokens:
            return torch.zeros_like(valid_pairs)
        cross = torch.as_tensor(
            chain_np[None, :] != chain_np[:, None],
            device=valid_pairs.device,
            dtype=torch.bool,
        )
    return valid_pairs & cross


def compute_ptm(
    pae: torch.Tensor,
    to_calculate: torch.Tensor | None,
    max_distance: float = 32,
    bin_count: int = 64,
):
    """Compute the predicted TM-score (PTM) from the predicted aligned error (PAE).

    Args:
        pae: Predicted aligned error tensor.
        to_calculate: Tensor indicating which residues to calculate PTM for.

    Returns:
        ptm: Computed predicted TM-score.
    """
    D, I = pae.shape[0], pae.shape[1]
    if to_calculate is None:
        to_calculate = torch.ones((I, I), dtype=torch.bool, device=pae.device)

    bin_centers = find_bin_midpoints(
        max_distance, bin_count, device=pae.device
    )  # TODO: get this from config
    pae = torch.nn.Softmax(dim=-1)(pae).detach().float()
    normalization_factor = 1.24 * (max(I, 19) - 15.0) ** (1 / 3) - 1.8
    denominator = 1 / (1 + (bin_centers / (normalization_factor)) ** 2)
    pae = pae * denominator[None, None, None, :]  # Broadcast to match pae shape

    pae = pae.sum(dim=-1)  # Sum over the last dimension
    pae = (pae * to_calculate[None]).sum(dim=-1) / (to_calculate.sum(dim=-1) + 1e-6)
    ptm = pae.max(dim=-1).values
    assert ptm.shape == (D,)
    return ptm


class ComputePTM(Metric):
    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "pae": ("network_output", "pae"),
            "asym_id": ("network_input", "f", "asym_id"),
        }

    def compute(
        self,
        pae: torch.Tensor,
        asym_id: torch.Tensor,
    ) -> dict[str, float]:
        """Compute the predicted TM-score (PTM) from the predicted aligned error (PAE).
        Args:
            pae: Predicted aligned error tensor.
            asym_id: AtomArrayStack containing the predicted structure.
        Returns:
            ptm: Computed predicted TM-score.
        """
        ptm = compute_ptm(pae, None)
        # split the batch dimension into separate keys in the output dictionary
        ptm = ptm.cpu().numpy()
        ptm = {f"ptm_{i}": ptm[i] for i in range(len(ptm))}
        return ptm


class ComputeIPTM(Metric):
    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "pae": ("network_output", "pae"),
            "asym_id": ("network_input", "f", "asym_id"),
            "is_ligand": ("network_input", "f", "is_ligand"),
        }

    def compute(
        self,
        pae: torch.Tensor,
        asym_id: torch.Tensor,
        is_ligand: torch.Tensor,
    ) -> dict[str, float]:
        """Compute the predicted interface TM-score (iPTM) from the predicted aligned error (PAE).
        Args:
            pae: Predicted aligned error tensor.
            predicted_atom_array_stack: AtomArrayStack containing the predicted structure.
        Returns:
            iptm: Computed interface TM-score.
        """
        # asym_id/is_ligand may arrive with a leading batch dim of 1 (e.g. from the
        # inference-engine's collated single-example batch) -- must be squeezed to a
        # plain (I,) vector before building (I, I) pairwise masks below, same as
        # cross_chain_pair_mask already does for chain_ids.
        while asym_id.ndim > 1 and asym_id.shape[0] == 1:
            asym_id = asym_id[0]
        while is_ligand.ndim > 1 and is_ligand.shape[0] == 1:
            is_ligand = is_ligand[0]

        unique, counts = torch.unique(asym_id, return_counts=True)
        to_calculate = asym_id[None, :] != asym_id[:, None]
        iptm = compute_ptm(pae, to_calculate)

        # make a protein - ligand mask
        protein_mask = is_ligand == 0
        ligand_mask = is_ligand == 1
        # calculate iptm for protein-protein, protein-ligand, and ligand-ligand interfaces
        protein_protein_mask = (
            protein_mask[None, :] & protein_mask[:, None] * to_calculate
        )
        protein_ligand_mask = (
            (protein_mask[None, :] & ligand_mask[:, None])
            | (ligand_mask[None, :] & protein_mask[:, None])
        ) * to_calculate
        ligand_ligand_mask = ligand_mask[None, :] & ligand_mask[:, None] * to_calculate
        # calculate iptm for each interface type
        iptm_protein_protein = compute_ptm(pae, protein_protein_mask)
        iptm_protein_ligand = compute_ptm(pae, protein_ligand_mask)
        iptm_ligand_ligand = compute_ptm(pae, ligand_ligand_mask)

        # split the batch dimension into separate keys in the output dictionary
        iptm = iptm.cpu().numpy()
        iptm = {f"iptm_{i}": iptm[i] for i in range(len(iptm))}
        iptm_protein_protein = iptm_protein_protein.cpu().numpy()
        iptm_protein_protein = {
            f"iptm_protein_protein_{i}": iptm_protein_protein[i]
            for i in range(len(iptm_protein_protein))
        }
        iptm_protein_ligand = iptm_protein_ligand.cpu().numpy()
        iptm_protein_ligand = {
            f"iptm_protein_ligand_{i}": iptm_protein_ligand[i]
            for i in range(len(iptm_protein_ligand))
        }
        iptm_ligand_ligand = iptm_ligand_ligand.cpu().numpy()
        iptm_ligand_ligand = {
            f"iptm_ligand_ligand_{i}": iptm_ligand_ligand[i]
            for i in range(len(iptm_ligand_ligand))
        }
        iptm.update(iptm_protein_protein)
        iptm.update(iptm_protein_ligand)
        iptm.update(iptm_ligand_ligand)
        return iptm


class ReportMeanPAEandAE(Metric):
    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "seq": ("extra_info", "seq"),
            "atom_frames": ("extra_info", "atom_frames"),
            "X_gt_L": ("extra_info", "X_gt_L"),
            "X_exists_L": ("extra_info", "crd_mask_L"),
            "X_pred_L": ("network_output", "X_pred_rollout_L"),
            "frame_atom_idxs": ("extra_info", "frame_atom_idxs"),
            "pae_logits": ("network_output", "pae"),
            "asym_id": ("network_input", "f", "asym_id"),
            "chain_iid_token_lvl": ("extra_info", "chain_iid_token_lvl"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["asym_id", "chain_iid_token_lvl"])

    def compute(
        self,
        seq: torch.Tensor,
        atom_frames: torch.Tensor,
        X_gt_L: torch.Tensor,
        X_exists_L: torch.Tensor,
        X_pred_L: torch.Tensor,
        frame_atom_idxs: torch.Tensor,
        pae_logits: torch.Tensor,
        asym_id: torch.Tensor | None = None,
        chain_iid_token_lvl: torch.Tensor | np.ndarray | list | None = None,
    ) -> dict[str, float]:
        """Report the mean aligned error and predicted aligned error (PAE) for each structure.
        Args:
            X_gt_L: Ground truth coords
            X_exists_L: Ground truth mask
            X_pred_L: Predicted coords
            frame_atom_idxs: Frame atom indices
            pae: Predicted aligned error tensor.
        Returns:
            pae_dict: Computed predicted aligned error per struct, and aligned error per struct.
        """
        B = pae_logits.shape[0]
        true_pae, true_pae_raw, valid_pae_pairs = self.calc_ae(
            seq,
            atom_frames,
            X_pred_L,
            X_gt_L,
            X_exists_L,
            frame_atom_idxs,
        )

        # fd WARN: these values should not be hardcoded
        pae_per_structure = unbin_logits(pae_logits.permute(0, 3, 1, 2), 32.0, 64)

        ae_pae_by_batch: dict[str, float] = {}
        for i in range(B):
            mask = valid_pae_pairs[i]
            ae_i = true_pae[i][mask]
            ae_i_raw = true_pae_raw[i][mask]
            pae_i = pae_per_structure[i][mask]
            err = pae_i - ae_i
            ae_pae_by_batch[f"mean_ae_{i}"] = ae_i.mean().item()
            # Uncapped true AE (Å) -- the actual docking/positional error, unlike mean_ae_{i}
            # which is clamped to 32 A to stay comparable with the model's PAE prediction
            # ceiling. Use this, not mean_ae_{i}, to answer "how far off is it really".
            ae_pae_by_batch[f"mean_ae_raw_{i}"] = ae_i_raw.mean().item()
            ae_pae_by_batch[f"mean_pae_{i}"] = pae_i.mean().item()
            # Pair-level calibration errors (Å / Å²) over valid token pairs.
            ae_pae_by_batch[f"mean_l1_pae_ae_{i}"] = err.abs().mean().item()
            ae_pae_by_batch[f"mean_l2_pae_ae_{i}"] = (err * err).mean().item()

            if chain_iid_token_lvl is not None or asym_id is not None:
                # Prefer asym_id (same as iPTM); fall back to token-level chain IIDs.
                chain_ids = asym_id if asym_id is not None else chain_iid_token_lvl
                cross_mask = cross_chain_pair_mask(chain_ids, mask)
                if cross_mask.any():
                    ae_cc = true_pae[i][cross_mask]
                    ae_cc_raw = true_pae_raw[i][cross_mask]
                    pae_cc = pae_per_structure[i][cross_mask]
                    ae_pae_by_batch[f"cross_chain_mean_ae_{i}"] = ae_cc.mean().item()
                    ae_pae_by_batch[f"cross_chain_mean_ae_raw_{i}"] = ae_cc_raw.mean().item()
                    ae_pae_by_batch[f"cross_chain_min_ae_raw_{i}"] = ae_cc_raw.min().item()
                    ae_pae_by_batch[f"cross_chain_max_ae_raw_{i}"] = ae_cc_raw.max().item()
                    ae_pae_by_batch[f"cross_chain_mean_pae_{i}"] = pae_cc.mean().item()
                    ae_pae_by_batch[f"cross_chain_min_ae_{i}"] = ae_cc.min().item()
                    ae_pae_by_batch[f"cross_chain_min_pae_{i}"] = pae_cc.min().item()
                else:
                    nan = float("nan")
                    ae_pae_by_batch[f"cross_chain_mean_ae_{i}"] = nan
                    ae_pae_by_batch[f"cross_chain_mean_ae_raw_{i}"] = nan
                    ae_pae_by_batch[f"cross_chain_min_ae_raw_{i}"] = nan
                    ae_pae_by_batch[f"cross_chain_max_ae_raw_{i}"] = nan
                    ae_pae_by_batch[f"cross_chain_mean_pae_{i}"] = nan
                    ae_pae_by_batch[f"cross_chain_min_ae_{i}"] = nan
                    ae_pae_by_batch[f"cross_chain_min_pae_{i}"] = nan

        return ae_pae_by_batch

    # fd Note that this duplicates logic from the loss fn. perhaps can be combined?
    def calc_ae(
        self,
        seq,
        atom_frames,
        X_pred_L,
        X_gt_L,
        X_exists_L,
        frame_atom_idxs,
        eps=1e-4,
    ):
        B = X_pred_L.shape[0]

        # Construct the backbone atoms in the faux atom-36 representation so we can use existing machinery to get frames
        frame_atom_idxs = frame_atom_idxs.unsqueeze(0).expand(B, -1, -1)
        X_pred_I = torch.zeros(B, seq.shape[-1], 36, 3, device=X_pred_L.device)
        X_pred_I[..., 0, :] = torch.gather(
            X_pred_L, 1, frame_atom_idxs[..., 0].unsqueeze(-1).expand(-1, -1, 3)
        )
        X_pred_I[..., 1, :] = torch.gather(
            X_pred_L, 1, frame_atom_idxs[..., 1].unsqueeze(-1).expand(-1, -1, 3)
        )
        X_pred_I[..., 2, :] = torch.gather(
            X_pred_L, 1, frame_atom_idxs[..., 2].unsqueeze(-1).expand(-1, -1, 3)
        )

        X_gt_I = torch.zeros(B, seq.shape[-1], 36, 3, device=X_gt_L.device)
        X_gt_I[..., 0, :] = torch.gather(
            X_gt_L, 1, frame_atom_idxs[..., 0].unsqueeze(-1).expand(-1, -1, 3)
        )
        X_gt_I[..., 1, :] = torch.gather(
            X_gt_L, 1, frame_atom_idxs[..., 1].unsqueeze(-1).expand(-1, -1, 3)
        )
        X_gt_I[..., 2, :] = torch.gather(
            X_gt_L, 1, frame_atom_idxs[..., 2].unsqueeze(-1).expand(-1, -1, 3)
        )

        atom_mask = torch.zeros(
            B, seq.shape[-1], 36, device=X_exists_L.device, dtype=torch.bool
        )
        atom_mask[..., 0] = torch.gather(X_exists_L, 1, frame_atom_idxs[..., 0])
        atom_mask[..., 1] = torch.gather(X_exists_L, 1, frame_atom_idxs[..., 1])
        atom_mask[..., 2] = torch.gather(X_exists_L, 1, frame_atom_idxs[..., 2])

        frames, frame_mask = get_frames(
            0,
            0,
            seq.unsqueeze(0).repeat(B, 1),
            frame_indices.to(seq.device),
            atom_frames,
        )

        N, L, natoms, _ = X_pred_I.shape

        # flatten middle dims so can gather across residues
        X_prime = X_pred_I.reshape(N, L * natoms, -1, 3).repeat(1, 1, NFRAMES, 1)
        Y_prime = X_gt_I.reshape(N, L * natoms, -1, 3).repeat(1, 1, NFRAMES, 1)
        frames_reindex_batched, frame_mask_batched = mask_unresolved_frames_batched(
            frames, frame_mask, atom_mask
        )

        X_x = torch.gather(
            X_prime, 1, frames_reindex_batched[..., 0:1].repeat(1, 1, 1, 3)
        )
        X_y = torch.gather(
            X_prime, 1, frames_reindex_batched[..., 1:2].repeat(1, 1, 1, 3)
        )
        X_z = torch.gather(
            X_prime, 1, frames_reindex_batched[..., 2:3].repeat(1, 1, 1, 3)
        )
        uX, tX = rigid_from_3_points(X_x, X_y, X_z)

        Y_x = torch.gather(
            Y_prime, 1, frames_reindex_batched[..., 0:1].repeat(1, 1, 1, 3)
        )
        Y_y = torch.gather(
            Y_prime, 1, frames_reindex_batched[..., 1:2].repeat(1, 1, 1, 3)
        )
        Y_z = torch.gather(
            Y_prime, 1, frames_reindex_batched[..., 2:3].repeat(1, 1, 1, 3)
        )
        uY, tY = rigid_from_3_points(Y_x, Y_y, Y_z)

        uX = uX[:, :, 0]
        uY = uY[:, :, 0]

        # Compute xij_ca across the batch
        # uX: (B, L, 3), X_pred_I: (B, A, 3), X_y: (B, L, 3)
        xij_ca = torch.einsum(
            "bfji,bfaj->bfai",
            uX,  # select valid frames for backbone, shape (B, N_valid_frames, 3)
            X_pred_I[:, None, :, 1] - X_y[:, :, None, 0],
        )  # Result: (B, N_valid_frames, N_valid_ca, 3)

        # Compute xij_ca_t across the batch
        # uY: (B, L, 3), X_gt_I: (B, A, 3), Y_y: (B, L, 3)
        xij_ca_t = torch.einsum(
            "bfji,bfaj->bfai",
            uY,  # select valid frames for backbone, shape (B, N_valid_frames, 3)
            X_gt_I[:, None, :, 1] - Y_y[:, :, None, 0],
        )  # Result: (B, N_valid_frames, N_valid_ca, 3)

        valid_frames = frame_mask_batched[:, :, 0]  # valid backbone frames (B,I)
        valid_ca = atom_mask[:, :, 1]  # valid CA atoms (B,I)
        valid_pairs = (
            valid_frames[:, :, None] & valid_ca[:, None, :]
        )  # valid pairs (B,I,I)

        eij_label_raw = torch.sqrt(torch.square(xij_ca - xij_ca_t).sum(dim=-1) + eps)
        # The model can never predict a PAE above its max bin (32 A, see the hardcoded
        # unbin_logits(..., 32.0, 64) call in compute() above) -- comparing that capped
        # prediction against an uncapped true AE systematically drags "bias" negative on
        # badly-wrong structures (true AE can be 40+ A, predicted PAE tops out at ~32 A).
        # Cap the true label the same way so bias reflects real mis-calibration, not this
        # ceiling mismatch. The uncapped value is also returned (eij_label_raw) for callers
        # that want the actual docking error, not the calibration-comparable one.
        eij_label = eij_label_raw.clamp(max=32.0)

        return eij_label, eij_label_raw, valid_pairs


class ReportMeanPDEandDE(Metric):
    """PDE analogue of ReportMeanPAEandAE: reports mean predicted distance error (PDE)
    and mean true distance error (DE) per structure, using the same representative-atom
    pairwise-distance definition as ConfidenceLoss.calc_pde (rf3/loss/af3_confidence_loss.py)."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "X_gt_L": ("extra_info", "X_gt_L"),
            "X_exists_L": ("extra_info", "crd_mask_L"),
            "X_pred_L": ("network_output", "X_pred_rollout_L"),
            "rep_atom_idxs": ("extra_info", "rep_atom_idxs"),
            "pde_logits": ("network_output", "pde"),
            "asym_id": ("network_input", "f", "asym_id"),
            "chain_iid_token_lvl": ("extra_info", "chain_iid_token_lvl"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["asym_id", "chain_iid_token_lvl"])

    def compute(
        self,
        X_gt_L: torch.Tensor,
        X_exists_L: torch.Tensor,
        X_pred_L: torch.Tensor,
        rep_atom_idxs: torch.Tensor,
        pde_logits: torch.Tensor,
        asym_id: torch.Tensor | None = None,
        chain_iid_token_lvl: torch.Tensor | np.ndarray | list | None = None,
    ) -> dict[str, float]:
        """Report the mean true distance error (DE) and predicted distance error (PDE)
        for each structure, over representative-atom pairs (same pairs the PDE head is
        trained against).
        """
        B = pde_logits.shape[0]
        true_de, valid_pde_pairs = self.calc_de(X_pred_L, X_gt_L, X_exists_L, rep_atom_idxs)

        # fd WARN: these values should not be hardcoded (matches ReportMeanPAEandAE)
        pde_per_structure = unbin_logits(pde_logits.permute(0, 3, 1, 2), 32.0, 64)

        de_pde_by_batch: dict[str, float] = {}
        for i in range(B):
            mask = valid_pde_pairs[i]
            de_i = true_de[i][mask]
            pde_i = pde_per_structure[i][mask]
            err = pde_i - de_i
            de_pde_by_batch[f"mean_de_{i}"] = de_i.mean().item()
            de_pde_by_batch[f"mean_pde_{i}"] = pde_i.mean().item()
            de_pde_by_batch[f"mean_l1_pde_de_{i}"] = err.abs().mean().item()
            de_pde_by_batch[f"mean_l2_pde_de_{i}"] = (err * err).mean().item()

            if chain_iid_token_lvl is not None or asym_id is not None:
                chain_ids = asym_id if asym_id is not None else chain_iid_token_lvl
                cross_mask = cross_chain_pair_mask(chain_ids, mask)
                if cross_mask.any():
                    de_cc = true_de[i][cross_mask]
                    pde_cc = pde_per_structure[i][cross_mask]
                    de_pde_by_batch[f"cross_chain_mean_de_{i}"] = de_cc.mean().item()
                    de_pde_by_batch[f"cross_chain_mean_pde_{i}"] = pde_cc.mean().item()
                    de_pde_by_batch[f"cross_chain_min_de_{i}"] = de_cc.min().item()
                    de_pde_by_batch[f"cross_chain_min_pde_{i}"] = pde_cc.min().item()
                else:
                    nan = float("nan")
                    de_pde_by_batch[f"cross_chain_mean_de_{i}"] = nan
                    de_pde_by_batch[f"cross_chain_mean_pde_{i}"] = nan
                    de_pde_by_batch[f"cross_chain_min_de_{i}"] = nan
                    de_pde_by_batch[f"cross_chain_min_pde_{i}"] = nan

        return de_pde_by_batch

    def calc_de(self, X_pred_L, X_gt_L, X_exists_L, rep_atom_idxs):
        X_pred_I = X_pred_L.index_select(1, rep_atom_idxs)
        X_gt_I = X_gt_L.index_select(1, rep_atom_idxs)
        X_exists_I = X_exists_L.index_select(1, rep_atom_idxs)
        predicted_distances = torch.cdist(
            X_pred_I, X_pred_I, compute_mode="donot_use_mm_for_euclid_dist"
        )
        ground_truth_distances = torch.cdist(
            X_gt_I, X_gt_I, compute_mode="donot_use_mm_for_euclid_dist"
        )
        de_label = torch.abs(ground_truth_distances - predicted_distances).detach()
        # Same ceiling-mismatch fix as ReportMeanPAEandAE.calc_ae: the model can never
        # predict a PDE above its max bin (32 A), so cap the true DE label the same way
        # or "bias" gets dragged negative purely by the ceiling, not real mis-calibration.
        de_label = de_label.clamp(max=32.0)
        valid_pairs = (X_exists_I.unsqueeze(-1) * X_exists_I.unsqueeze(-2)).detach()
        return de_label, valid_pairs
