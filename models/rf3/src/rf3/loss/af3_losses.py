import numpy as np
import torch
import torch.nn as nn

from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.alignment import weighted_rigid_align


# resolve residue-level symmetries in native vs pred
class ResidueSymmetryResolution(nn.Module):
    def _get_best(self, x_pred, x_native, x_native_mask, a_i):
        mask = torch.zeros_like(x_native_mask[0])
        mask[a_i[0]] = True
        d_pred = torch.cdist(x_pred[:, mask], x_pred[:, ~mask])
        x_nat_j = x_native.clone()
        for j in range(a_i.shape[0]):
            x_nat_j[:, a_i[0]] = x_native[:, a_i[j]]
            d_nat = torch.cdist(x_nat_j[:, mask], x_nat_j[:, ~mask])
            drms_j = torch.square(d_pred - d_nat).nan_to_num()
            drms_j[drms_j > 15] = 15
            drms_j = torch.mean(drms_j, dim=(-1, -2))
            if j == 0:
                bestj = torch.zeros(
                    x_pred.shape[0], dtype=torch.long, device=x_pred.device
                )
                bestrms = drms_j
            else:
                bestj[drms_j < bestrms] = j
                bestrms[drms_j < bestrms] = drms_j[drms_j < bestrms]
        # x_nat_j[:,a_i[0]] = x_native[:,a_i[j]]
        for j in range(x_pred.shape[0]):
            x_native[j, a_i[0]] = x_native[j, a_i[bestj[j]]]
            x_native_mask[j, a_i[0]] = x_native_mask[j, a_i[bestj[j]]]

        return x_native, x_native_mask

    def forward(self, x_pred, x_native, x_native_mask, automorph_input):
        assert isinstance(
            automorph_input, list
        ), "automorph_input should be a list of arrays"

        x_native_best = []
        x_native_mask_best = []
        for batch_index, automorph_input_per_batch in enumerate(automorph_input):
            for a_i in automorph_input_per_batch:
                if a_i.shape[0] == 1:
                    continue
                a_i = torch.tensor(a_i, device=x_pred.device)
                x_native, x_native_mask = self._get_best(
                    x_pred[batch_index],
                    x_native[batch_index],
                    x_native_mask[batch_index],
                    a_i,
                )
                # this function outputs unbatched tensors
                x_native = x_native[None]
                x_native_mask = x_native_mask[None]
            x_native_best.append(x_native)
            x_native_mask_best.append(x_native_mask)

        if len(x_native_best) > 1:
            x_native = torch.stack(x_native_best, dim=0)
            x_native_mask = torch.stack(x_native_mask_best, dim=0)
        else:
            x_native = x_native_best[0]
            x_native_mask = x_native_mask_best[0]

        return x_native,x_native_mask


# Resolve subunit-level symmetries in native vs pred
class SubunitSymmetryResolution(nn.Module):
    def __init__(self, **losses):
        super().__init__()

    def _rms_align(self, X_fixed, X_moving):
        # input:
        #   X_fixed = predicted = Nbatch x L x 3
        #   X_moving = native = Nambig x L x 3
        # output:
        #   X_pre = Nambig x Nbatch x 3
        #   U = Nambig x Nbatch x 3 x 3
        #   X_post = Nambig x Nbatch x 3
        assert X_fixed.shape[-2:] == X_moving.shape[-2:]
        Nbatch = X_fixed.shape[0]
        Nambig = X_moving.shape[0]
        X_fixed = X_fixed[None, :]
        X_moving = X_moving[:, None]

        u_X_fixed = torch.mean(X_fixed, dim=-2)
        u_X_moving = torch.mean(X_moving, dim=-2)

        X_fixed = X_fixed - u_X_fixed.unsqueeze(-2)
        X_moving = X_moving - u_X_moving.unsqueeze(-2)

        # force f32
        C = torch.einsum("...ji,...jk->...ik", X_moving.float(), X_fixed.float())
        # Guard against degenerate/non-finite C, which can hang cuSOLVER's SVD.
        C = torch.nan_to_num(C)
        scale = C.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1)  # per-matrix magnitude
        ridge = (1e-4 * scale).clamp_min(1e-6)[..., None, None]
        C = C + ridge * torch.eye(3, device=C.device, dtype=C.dtype)
        U, S, V = torch.linalg.svd(C.float())
        R = U @ V
        F = torch.eye(3, 3, device=X_fixed.device)[None, None].repeat(
            Nambig, Nbatch, 1, 1
        )
        F[..., -1, -1] = torch.sign(torch.linalg.det(R.float()))
        R = U @ F @ V
        return u_X_moving, R, u_X_fixed

    def _greedy_resolve_mapping(
        self,
        dist,
        iid_to_index,
        entity_to_index,
        iids_by_entity,
        entity_by_iids,
        nmodel_by_iid,
    ):
        # returns:
        #    best_xform      tensor [i]->transform number
        #    best_assignment dict{pred_iid:[native_iids]} (batch)
        nTransforms = dist.shape[0]
        nIid = dist.shape[1]
        nBatch = dist.shape[-1]
        toAssign = [k for k, v in nmodel_by_iid.items() if int(v) > 0]

        # sort equiv groups by # resolved residues
        # first make that list
        nmodel_by_equiv = {
            int(i): 0 for i in entity_to_index.keys()
        }  # torch.zeros(nEquiv,dtype=torch.long,device=dist.device)
        for i, iid in enumerate(toAssign):
            nmodel_by_equiv[entity_by_iids[iid]] += nmodel_by_iid[iid]
        equiv_order = sorted(
            nmodel_by_equiv, key=nmodel_by_equiv.get
        )  # torch.argsort(nmodel_by_equiv,descending=True)

        best_cost = torch.zeros(nBatch, device=dist.device)
        best_xform = torch.zeros(nBatch, dtype=torch.long, device=dist.device)
        best_assignment = {
            int(i): torch.zeros(nBatch, dtype=torch.long, device=dist.device)
            for i in toAssign
        }
        for t in range(nTransforms):
            # then sort with most res first
            cost = torch.zeros(nBatch, device=dist.device)
            assignment = {
                int(i): torch.full(
                    (nBatch,), int(i), dtype=torch.long, device=dist.device
                )
                for i in toAssign
            }

            for i_equiv in equiv_order:
                mask_equiv = torch.zeros(
                    (nIid, nIid), dtype=torch.bool, device=dist.device
                )
                iids_in_i_equiv = iids_by_entity[i_equiv]
                nIids_in_i_equiv = iids_in_i_equiv.shape[0]
                iid_idxs_in_i_equiv = np.vectorize(iid_to_index.__getitem__)(
                    iids_in_i_equiv
                )

                # Only chains with cropped/modelled atoms participate. Including
                # zero-modelled iids lets argmin pick a pred iid absent from
                # ``assignment`` (keys = toAssign) and raise KeyError.
                modelled_local = [
                    int(i)
                    for i in iids_in_i_equiv
                    if int(nmodel_by_iid[int(i)]) > 0
                ]
                nResolvedEntities_i = len(modelled_local)
                if nResolvedEntities_i == 0:
                    continue

                mask_equiv[
                    iid_idxs_in_i_equiv[:, None], iid_idxs_in_i_equiv[None, :]
                ] = True
                wted_dist = (
                    dist[t, mask_equiv]
                    .nan_to_num(1e9)
                    .view(nIids_in_i_equiv, nIids_in_i_equiv, nBatch)
                    .clone()
                )
                unmodelled = torch.tensor(
                    [int(nmodel_by_iid[int(i)]) <= 0 for i in iids_in_i_equiv],
                    dtype=torch.bool,
                    device=dist.device,
                )
                if unmodelled.any():
                    wted_dist[unmodelled, :, :] = 1e9
                    wted_dist[:, unmodelled, :] = 1e9

                # greedily assign min RMS within each equiv group
                for _ in range(nResolvedEntities_i):
                    flat = wted_dist.view(
                        nIids_in_i_equiv * nIids_in_i_equiv, nBatch
                    )
                    pn = torch.argmin(flat, dim=0)

                    # special case: if there is NO seq overlap between predicted and native peptides,
                    # fall back to identity assignment
                    if (flat[pn, torch.arange(nBatch, device=flat.device)] >= 1e9).all():
                        break

                    i_nat = pn // nIids_in_i_equiv
                    i_pred = pn % nIids_in_i_equiv
                    # weight the total cost by #residues on the assigned pred chain
                    for j, (ii_nat, ii_pred) in enumerate(zip(i_nat, i_pred)):
                        pred_iid = int(iids_by_entity[int(i_equiv)][int(ii_pred)])
                        nat_iid = int(iids_by_entity[int(i_equiv)][int(ii_nat)])
                        if pred_iid not in assignment:
                            # Defensive: should be unreachable after unmodelled masking.
                            continue
                        cost[j] = cost[j] + flat[pn[j], j] * int(
                            nmodel_by_iid[pred_iid]
                        )
                        assignment[pred_iid][j] = nat_iid

                    for j in range(i_nat.shape[0]):
                        wted_dist[i_nat[j], :, j] = 1e6
                        wted_dist[:, i_pred[j], j] = 1e6
            if t == 0:
                best_cost = cost
                best_assignment = assignment
            else:
                mask = cost < best_cost
                best_cost[mask] = cost[mask]
                for i, bi in best_assignment.items():
                    best_assignment[i][mask] = assignment[i][mask]
                best_xform[mask] = t

        return (best_xform, best_assignment)

    def _resolve_subunits(
        self, mol_entities, mol_iid, crop_mask, x_native, mask_native, x_pred, is_padding_L, tag
    ):
        Nbatch = x_pred.shape[0]

        # index -> entity
        all_entities = torch.unique(mol_entities)
        entity_to_index = {int(ii): i for i, ii in enumerate(all_entities)}

        # index -> iid
        all_iids = torch.unique(mol_iid).cpu().numpy()
        Niids = len(all_iids)
        iid_to_index = {int(ii): i for i, ii in enumerate(all_iids)}

        # entity -> iid list
        iids_by_entity = {
            int(i): torch.unique(mol_iid[mol_entities == i]).long().cpu().numpy()
            for i in all_entities
        }

        # iid -> entity list
        entity_by_iids = {
            int(i): torch.unique(mol_entities[mol_iid == i]).long().cpu().item()
            for i in all_iids
        }

        # Fix mismatched chain lengths within entities by splitting into separate entities
        next_entity_id = int(all_entities.max().item()) + 1
        reassigned = {}  # iid -> new entity

        for entity, iids in iids_by_entity.items():
            lengths = {int(iid): (mol_iid == iid).sum().item() for iid in iids}
            if len(set(lengths.values())) > 1:
                print(f"WARNING: {tag} entity {entity} has iids with mismatched lengths: {lengths}")
                length_to_iids = {}
                for iid, length in lengths.items():
                    length_to_iids.setdefault(length, []).append(iid)

                # Keep the largest group as the original entity, reassign others
                largest_group = max(length_to_iids.values(), key=len)
                for length, group_iids in length_to_iids.items():
                    if group_iids == largest_group:
                        continue
                    for iid in group_iids:
                        reassigned[iid] = next_entity_id
                    next_entity_id += 1

        # Apply reassignments
        if reassigned:
            # Rebuild mol_entities (local copy) with new entity ids
            mol_entities = mol_entities.clone()
            for iid, new_entity in reassigned.items():
                mol_entities[mol_iid == iid] = new_entity

            # Rebuild all derived mappings
            all_entities = torch.unique(mol_entities)
            entity_to_index = {int(ii): i for i, ii in enumerate(all_entities)}
            iids_by_entity = {
                int(i): torch.unique(mol_iid[mol_entities == i]).long().cpu().numpy()
                for i in all_entities
            }
            entity_by_iids = {
                int(i): torch.unique(mol_entities[mol_iid == i]).long().cpu().item()
                for i in all_iids
            }
        # 1) get the iid with most resolved residues
        mask = torch.zeros(
            mol_entities.shape[0], dtype=torch.bool, device=mol_iid.device
        )
        # crop mask is padded
        mask[crop_mask[crop_mask>=0]] = 1

        mask_by_iid = {int(i): mask[mol_iid == i] for i in all_iids}
        mask_native_by_iid = {int(i): mask_native[mol_iid == i] for i in all_iids}
        nmodeled_by_iid = {
            int(i): torch.sum(mask_by_iid[i]) for i in mask_native_by_iid.keys()
        }

        iid_src_idx = max(nmodeled_by_iid, key=nmodeled_by_iid.get)
        entity_src_idx = entity_by_iids[iid_src_idx]
        native_by_iid = {int(i): x_native[mol_iid == i] for i in all_iids}
        # crop mask is padded
        pred_by_iid = {int(ii): x_pred[:, (mol_iid[crop_mask] == ii)*(crop_mask>=0)] for ii in all_iids}

        # align it to all equivalent targets
        equiv_native_iids = iids_by_entity[entity_src_idx]

        xpres, Us, xposts = [], [], []

        for n in equiv_native_iids:
            nat_n = native_by_iid[int(n)][mask_by_iid[int(iid_src_idx)]]
            pred_n = pred_by_iid[int(iid_src_idx)]
            mask_unres = ~nat_n[..., 0].isnan()
            nat_n = nat_n[mask_unres]
            pred_n = pred_n[:, mask_unres]

            if mask_unres.sum() > 3:
                xpre, U, xpost = self._rms_align(pred_n, nat_n[None])
                xpres.append(xpre)
                Us.append(U)
                xposts.append(xpost)

        xpres, Us, xposts = (
            torch.cat(xpres, dim=0),
            torch.cat(Us, dim=0),
            torch.cat(xposts, dim=0),
        )

        # ===== OPTIMIZED COM COMPUTATION =====
        # Build entity compatibility matrix: which iid pairs are compatible
        # Vectorized version - much faster for large numbers of iids
        entity_compat = torch.zeros((Niids, Niids), dtype=torch.bool, device=Us.device)

        # Create a mapping from iid_index to entity_index
        iid_to_entity_idx = torch.zeros(Niids, dtype=torch.long, device=Us.device)
        for iid in all_iids:
            iid_to_entity_idx[iid_to_index[int(iid)]] = entity_to_index[
                entity_by_iids[int(iid)]
            ]

        # Vectorized: two iids are compatible if they have the same entity
        entity_compat = iid_to_entity_idx.unsqueeze(0) == iid_to_entity_idx.unsqueeze(1)

        # Precompute all masks at once
        # Create tensors for vectorized operations
        max_residues = max(native_by_iid[int(i)].shape[0] for i in all_iids)

        # Pad native coords and masks to same length for vectorization
        native_coords_padded = torch.full(
            (Niids, max_residues, 3), float("nan"), device=Us.device
        )
        native_valid_padded = torch.zeros(
            (Niids, max_residues), dtype=torch.bool, device=Us.device
        )
        crop_mask_padded = torch.zeros(
            (Niids, max_residues), dtype=torch.bool, device=Us.device
        )

        for iid in all_iids:
            idx = iid_to_index[int(iid)]
            coords = native_by_iid[int(iid)]
            n_res = coords.shape[0]
            native_coords_padded[idx, :n_res] = coords
            native_valid_padded[idx, :n_res] = ~coords[:, 0].isnan()
            crop_mask_padded[idx, :n_res] = mask_by_iid[int(iid)]

        # Compute nat_com vectorized - optimized
        nat_com = torch.full((Niids, Niids, 3), float("nan"), device=Us.device)

        # Pre-create constants to avoid repeated tensor creation
        zero_tensor = torch.tensor(0.0, device=Us.device)

        for i in range(Niids):
            # Get valid mask for this iid across all potential j's
            valid_i = native_valid_padded[i]  # [max_residues]
            coords_i = native_coords_padded[i]  # [max_residues, 3]

            # Find which j's are compatible
            compatible_j = torch.where(entity_compat[i])[0]

            if len(compatible_j) > 0:
                # Vectorize across all compatible j's
                crop_masks_j = crop_mask_padded[
                    compatible_j
                ]  # [n_compat, max_residues]
                combined_masks = crop_masks_j & valid_i.unsqueeze(
                    0
                )  # [n_compat, max_residues]

                # Compute means where mask is valid
                mask_counts = combined_masks.sum(dim=1, keepdim=True)  # [n_compat, 1]
                valid_means = mask_counts.squeeze(-1) > 0  # [n_compat]

                # Compute COMs
                masked_coords = torch.where(
                    combined_masks.unsqueeze(-1), coords_i.unsqueeze(0), zero_tensor
                )  # [n_compat, max_residues, 3]

                sums = masked_coords.sum(dim=1)  # [n_compat, 3]
                means = sums / mask_counts.clamp(min=1)  # [n_compat, 3]

                # Assign back - only assign valid means, leave rest as NaN
                if valid_means.any():
                    nat_com[i, compatible_j[valid_means]] = means[valid_means]

        # Compute pred_com - with vectorized grouping
        pred_com = torch.full((Niids, Niids, Nbatch, 3), float("nan"), device=Us.device)

        # Get all compatible (i, j) pairs
        i_indices, j_indices = torch.where(entity_compat)

        # Sort by j to group all i's for each j together
        sorted_indices = torch.argsort(j_indices)
        j_indices_sorted = j_indices[sorted_indices]
        i_indices_sorted = i_indices[sorted_indices]

        # Find boundaries where j changes
        unique_j, counts = torch.unique_consecutive(
            j_indices_sorted, return_counts=True
        )

        # Create splits
        j_splits = torch.cumsum(counts, dim=0)
        j_starts = torch.cat([torch.tensor([0], device=Us.device), j_splits[:-1]])

        # Process each j with all its compatible i's at once
        for idx in range(len(unique_j)):
            j_idx = unique_j[idx].item()
            j = int(all_iids[j_idx])

            if j not in pred_by_iid:
                continue

            # Get all i indices for this j
            start = j_starts[idx].item()
            end = j_splits[idx].item()
            i_idx_tensor = i_indices_sorted[start:end]

            n_res_j = mask_by_iid[j].shape[0]
            crop_mask_j = crop_mask_padded[j_idx, :n_res_j]  # [n_res_j]

            # Get all native_valid masks for these i's at once
            native_valid_i_batch = native_valid_padded[
                i_idx_tensor, :n_res_j
            ]  # [n_i, n_res_j]

            # Compute overlap for all i's at once
            overlap_full_batch = (
                crop_mask_j.unsqueeze(0) & native_valid_i_batch
            )  # [n_i, n_res_j]

            # Find which i's have any overlap
            has_overlap = overlap_full_batch.any(dim=1)  # [n_i]

            # Filter to only i's with overlap
            valid_i_indices = i_idx_tensor[has_overlap]
            overlap_full_valid = overlap_full_batch[has_overlap]  # [n_valid, n_res_j]

            # Convert overlap masks to pred_by_iid space
            overlap_pred_space = overlap_full_valid[:, crop_mask_j]  # [n_valid, n_crop]

            # Check which ones actually have overlap in pred space
            has_overlap_pred = overlap_pred_space.any(dim=1)

            # Get final valid indices
            final_i_indices = valid_i_indices[has_overlap_pred]
            final_masks = overlap_pred_space[has_overlap_pred]  # [n_final, n_crop]

            # Get pred coords for this j: [Nbatch, n_crop, 3]
            pred_coords_j = pred_by_iid[j]

            # For each final i, compute mean
            # Expand pred_coords to [n_final, Nbatch, n_crop, 3]
            pred_coords_expanded = pred_coords_j.unsqueeze(0).expand(
                len(final_i_indices), -1, -1, -1
            )

            # Expand masks to [n_final, Nbatch, n_crop, 1]
            masks_expanded = (
                final_masks[:,None,:,None].expand(-1, Nbatch, -1, 3)
            )

            # Compute masked sum and count
            masked_coords = torch.where(
                masks_expanded,
                pred_coords_expanded,
                torch.tensor(0.0, device=Us.device),
            )
            sums = masked_coords.sum(dim=2)  # [n_final, Nbatch, 3]
            counts = final_masks.sum(dim=1, keepdim=True).unsqueeze(
                -1
            )  # [n_final, 1, 1]

            # Compute means
            means = sums / counts.clamp(min=1)  # [n_final, Nbatch, 3]

            # Assign to pred_com
            for k, i_idx in enumerate(final_i_indices):
                pred_com[i_idx.item(), j_idx, :, :] = means[k]

        # Continue with rest of the function...
        # apply all transforms to native
        nat_com_transformed = (
            torch.einsum(
                "ijkx,ijlxy->ijkly",
                nat_com[None, :, :, :] - xpres[:, None, :, :],
                Us[:, None],
            )
            + xposts[:, None, None]
        )

        # collect all distances
        dist = torch.linalg.norm(pred_com[None, :, :] - nat_com_transformed, dim=-1)

        # solve mapping (identity fallback if greedy assign fails on rare crops)
        try:
            transforms, assignment = self._greedy_resolve_mapping(
                dist,
                iid_to_index,
                entity_to_index,
                iids_by_entity,
                entity_by_iids,
                nmodeled_by_iid,
            )
        except Exception as e:
            print(
                f"WARNING: {tag} greedy subunit mapping failed "
                f"({type(e).__name__}: {e}); using identity assignment"
            )
            transforms = torch.zeros(
                Nbatch, dtype=torch.long, device=x_pred.device
            )
            assignment = {
                int(i): torch.full(
                    (Nbatch,), int(i), dtype=torch.long, device=x_pred.device
                )
                for i in all_iids
                if int(nmodeled_by_iid[int(i)]) > 0
            }

        # generate output stack
        x_native_aln = torch.zeros_like(x_pred)
        x_native_mask = torch.zeros(
            x_pred.shape[:2], dtype=torch.bool, device=x_pred.device
        )
        for i, si in assignment.items():
            for t in range(x_native_aln.shape[0]):
                mask_src = mol_iid == i
                combined_mask = torch.zeros(is_padding_L.shape[0], dtype=torch.bool, device=is_padding_L.device)
                combined_mask[~is_padding_L] = mask_src[mask]
                x_native_aln[t, combined_mask] = native_by_iid[int(si[t])][
                    mask_by_iid[int(i)]
                ].to(x_native_aln.dtype)
                x_native_mask[t, combined_mask] = mask_native_by_iid[int(si[t])][
                    mask_by_iid[int(i)]
                ].to(x_native_mask.dtype)

        return (x_native_aln, x_native_mask)

    def forward(self, mol_entities, mol_iid, mask_crop, x_native, mask_native, x_pred, is_padding_L, tag=''):
        B = x_pred.shape[0]

        x_native_aln, x_native_mask = [], []
        for b in range(B):
            x_native_aln_b, x_native_mask_b = self._resolve_subunits(
                mol_entities[b], mol_iid[b], mask_crop[b], x_native[b], mask_native[b], x_pred[b], is_padding_L[b], tag
            )
            x_native_aln.append(x_native_aln_b)
            x_native_mask.append(x_native_mask_b)

        x_native_aln = torch.stack(x_native_aln, dim=0)
        x_native_mask = torch.stack(x_native_mask, dim=0)

        return x_native_aln, x_native_mask


class ProteinLigandBondLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, network_input, network_output, loss_input):
        # find p/l bonds at token level
        is_ligand = network_input["f"]["is_ligand"]
        is_inter_polymer_ligand = torch.outer(is_ligand, ~is_ligand)
        token_bonds = network_input["f"]["token_bonds"]
        pl_bonds = token_bonds * is_inter_polymer_ligand
        first_tok, second_tok = pl_bonds.nonzero(as_tuple=True)

        # early exit
        if first_tok.numel() == 0:
            return torch.tensor(0.0), {"protein_ligand_bond_loss": torch.tensor(0.0)}

        # map tokens to atom level
        atom2token = network_input["f"]["atom_to_token_map"]
        pl_atoms = torch.zeros(
            (1, atom2token.shape[0], atom2token.shape[0]),
            dtype=torch.bool,
            device=atom2token.device,
        )
        for i, j in zip(first_tok, second_tok):
            pl_atoms += (atom2token == i)[None, :, None] * (atom2token == j)[
                None, None, :
            ]

        crd_mask_LL = (
            loss_input["crd_mask_L"][:, None] * loss_input["crd_mask_L"][:, :, None]
        )
        resolved_bonds = pl_atoms * crd_mask_LL

        # the mask may be different for each structure in the batch, so resolve bonds at the per-batch level
        b, atom1, atom2 = resolved_bonds.nonzero(as_tuple=True)

        # get loss
        X_L = network_output["X_L"]
        X_gt_L = loss_input["X_gt_L"]
        predicted_distances = torch.linalg.norm(X_L[b, atom1] - X_L[b, atom2], dim=-1)
        ground_truth_distances = torch.linalg.norm(
            X_gt_L[b, atom1] - X_gt_L[b, atom2], dim=-1
        )
        mask_bonded = ground_truth_distances < 2.4
        loss = torch.mean(
            torch.square(
                predicted_distances[mask_bonded] - ground_truth_distances[mask_bonded]
            )
        )

        return self.weight * loss, {"protein_ligand_bond_loss": loss.detach()}


def _expand_hotspot_mask(X_gt, hotspot_mask, crd_mask, cutoff=4.5):
    """Expand hotspot atom mask to include all atoms within `cutoff` Angstroms.

    Args:
        X_gt: [B, L, 3] ground truth coordinates.
        hotspot_mask: [B, L] boolean mask of hotspot-annotated atoms.
        crd_mask: [B, L] boolean mask of valid (resolved) coordinates.
        cutoff: distance threshold in Angstroms.

    Returns:
        [B, L] boolean mask — True for any valid atom within cutoff of a hotspot atom.
    """
    X_clean = torch.nan_to_num(X_gt, nan=1e6)
    dists = torch.cdist(X_clean, X_clean)  # [B, L, L]
    # Mask columns that are not hotspot atoms or not valid
    col_mask = hotspot_mask & crd_mask  # [B, L]
    dists = dists.masked_fill(~col_mask.unsqueeze(1), float("inf"))  # [B, L, L]
    min_dist = dists.min(dim=-1).values  # [B, L]
    return (min_dist < cutoff) & crd_mask


class DiffusionLoss(nn.Module):
    def __init__(
        self,
        weight,
        smoothed_lddt_weight,
        sigma_data,
        alpha_dna,
        alpha_rna,
        alpha_ligand,
        edm_lambda,
        se3_invariant_loss,
        alpha_hotspot_lddt=0.0,
        alpha_hotspot_diffusion=0.0,
    ):
        super().__init__()
        self.weight = weight
        self.smoothed_lddt_weight = smoothed_lddt_weight
        self.sigma_data = sigma_data
        self.alpha_dna = alpha_dna
        self.alpha_rna = alpha_rna
        self.alpha_ligand = alpha_ligand
        self.alpha_hotspot_lddt = alpha_hotspot_lddt
        self.alpha_hotspot_diffusion = alpha_hotspot_diffusion
        if edm_lambda:
            # original EDM scaling factor
            self.get_lambda = (
                lambda sigma: (sigma**2 + self.sigma_data**2)
                / (sigma * self.sigma_data) ** 2
            )
        else:
            # AF3 uses a weird scaling factor for their loss
            self.get_lambda = (
                lambda sigma: (sigma**2 + self.sigma_data**2)
                / (sigma + self.sigma_data) ** 2
            )
        self.se3_invariant_loss = se3_invariant_loss

    def forward(self, network_input, network_output, loss_input):
        X_L = network_output["X_L"]  # D, L, 3
        B, D = X_L.shape[0], X_L.shape[1]
        X_gt_L = loss_input["X_gt_L"]
        crd_mask_L = loss_input["crd_mask_L"]
        tok_idx = network_input["f"]["atom_to_token_map"]
        t = network_input["t"]  # (D,)

        w_L = 1 + (
            network_input["f"]["is_dna"] * self.alpha_dna
            + network_input["f"]["is_rna"] * self.alpha_rna
            + network_input["f"]["is_ligand"] * self.alpha_ligand
        ).gather(1, tok_idx.clamp(min=0).long())
        w_L = w_L[:, None].expand(-1, D, -1) * crd_mask_L
        # Hotspot diffusion weighting: upweight atoms within 4.5A of hotspot atoms
        if self.alpha_hotspot_diffusion > 0:
            hotspot_mask = network_input["f"]["mask_hotspot_1_atom"]  # [B, L]
            near_hotspot = _expand_hotspot_mask(
                X_gt_L[:, 0], hotspot_mask, crd_mask_L[:, 0], cutoff=4.5
            )  # [B, L]
            w_hotspot = self.alpha_hotspot_diffusion * near_hotspot.float()
            w_L = w_L + w_hotspot[:, None].expand(-1, D, -1) * crd_mask_L
        # TODO: also multiply by is_padding_L
        X_gt_aligned_L_batch = torch.zeros_like(X_L).to(X_L.device)
        for i in range(B):
            X_gt_aligned_L = weighted_rigid_align(
                X_L[i], X_gt_L[i], crd_mask_L[i, 0], w_L[i]
            )
            X_gt_aligned_L = torch.nan_to_num(X_gt_aligned_L)
            X_gt_aligned_L_batch[i] = X_gt_aligned_L
        l_mse = (
            1
            / 3
            * torch.div(
                torch.sum(w_L * torch.sum((X_L - X_gt_aligned_L_batch) ** 2, dim=-1), dim=-1),
                torch.sum(crd_mask_L, dim=-1) + 1e-4,
            )
        )  # w_L is already updated by the mask

        assert l_mse.shape == (
            B,
            D,
        )
        l_diffusion = self.get_lambda(t) * l_mse

        l_diffusion_total = torch.mean(l_diffusion)
        # smoothed lddt loss
        if self.alpha_hotspot_lddt > 0:
            pair_weight_atom = self.alpha_hotspot_lddt * network_input["f"]["mask_hotspot_1_atom"].float()
        else:
            pair_weight_atom = None
        smoothed_lddt_loss_ = smoothed_lddt_loss(
            X_L,
            X_gt_L,
            crd_mask_L,
            network_input["f"]["is_dna"],
            network_input["f"]["is_rna"],
            tok_idx,
            pair_weight_atom=pair_weight_atom,
        )
        l_diffusion_total += self.smoothed_lddt_weight * smoothed_lddt_loss_.mean()
        loss_dict = {
            "diffusion_loss": l_diffusion.detach(),
            "smoothed_lddt_loss": smoothed_lddt_loss_.detach(),
            "t": t.detach(),
        }

        return self.weight * l_diffusion_total, loss_dict

def smoothed_lddt_loss(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps=1e-6, pair_weight_atom=None):
    if pair_weight_atom is None:
        pair_weight_atom = torch.zeros(X_L.shape[0], X_L.shape[2], device=X_L.device)

    @activation_checkpointing
    def _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, pair_weight_atom, eps, use_amp=True):
        B, D, L = X_L.shape[:3]
        first_index, second_index = torch.triu_indices(L, L, 1, device=X_L.device)

        # compute the unique distances between all pairs of atoms
        X_gt_L = X_gt_L.nan_to_num()

        # only use native 1 (assumes dist map identical btwn all copies)
        ground_truth_distances = torch.linalg.norm(
            X_gt_L[:, 0:1, first_index] - X_gt_L[:, 0:1, second_index], dim=-1
        )
        tok_idx_valid = tok_idx.clamp(min=0)
        # only score pairs that are close enough in the ground truth
        if is_dna.any() or is_rna.any():
            is_na_L = is_dna.gather(
                1, tok_idx_valid[:, first_index].long()
            ) | is_rna.gather(1, tok_idx_valid[:, first_index].long())
            is_na_L = is_na_L[:, None]
        else:
            is_na_L = torch.zeros_like(ground_truth_distances, dtype=torch.bool)
        assert is_na_L.shape == ground_truth_distances.shape

        pair_mask = torch.logical_and(
            ground_truth_distances > 0,
            ground_truth_distances < torch.where(is_na_L, 30.0, 15.0),
        )
        del is_na_L

        # only score pairs that are resolved in the ground truth
        pair_mask *= crd_mask_L[:, 0:1, first_index] * crd_mask_L[:, 0:1, second_index]
        # don't score pairs that are in the same token
        pair_mask *= (
            tok_idx_valid[:, None, first_index] != tok_idx_valid[:, None, second_index]
        )

        _, _, valid_pairs = pair_mask.nonzero(as_tuple=True)
        pair_mask = pair_mask[:, :, valid_pairs].to(X_L.dtype)
        ground_truth_distances = ground_truth_distances[:, :, valid_pairs]
        first_index, second_index = first_index[valid_pairs], second_index[valid_pairs]

        # Hotspot pair weighting: upweight pairs where at least one atom is a hotspot
        w_first = pair_weight_atom[:, first_index].unsqueeze(1)   # [B, 1, N_valid]
        w_second = pair_weight_atom[:, second_index].unsqueeze(1)  # [B, 1, N_valid]
        pair_weight = pair_mask * (1.0 + torch.max(w_first, w_second))  # [B, 1, N_valid]

        predicted_distances = torch.linalg.norm(
            X_L[:, :, first_index] - X_L[:, :, second_index], dim=-1
        )

        delta_distances = torch.abs(predicted_distances - ground_truth_distances + eps)
        del predicted_distances, ground_truth_distances
        lddt = (
            0.25
            * (
                torch.sum(torch.sigmoid(0.5 - delta_distances) * pair_weight, dim=(-1))
                + torch.sum(torch.sigmoid(1.0 - delta_distances) * pair_weight, dim=(-1))
                + torch.sum(torch.sigmoid(2.0 - delta_distances) * pair_weight, dim=(-1))
                + torch.sum(torch.sigmoid(4.0 - delta_distances) * pair_weight, dim=(-1))
            )
            / (torch.sum(pair_weight, dim=(-1)) + eps)
        )

        return 1 - lddt

    return _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, pair_weight_atom, eps)


def distogram_loss(
    pred_distogram,
    X_rep_atoms_I,
    crd_mask_rep_atoms_I,
    cce_loss,
    min_distance=2,
    max_distance=22,
    bins=64,
    pair_weight=None,
):
    """
    computes distogram loss

    pair_weight: optional [B, I, I] tensor of per-pair multiplicative weights.
    When None, all valid pairs are weighted equally (current behavior).
    """
    distance_map = torch.cdist(X_rep_atoms_I, X_rep_atoms_I)
    distance_map[distance_map.isnan()] = 9999.0
    bins = torch.linspace(min_distance, max_distance, bins).to(X_rep_atoms_I.device)
    # Note that torch.bucketize adds a catch-all bin for values outside the range,
    # so we end up with n_bins + 1 bins (65 in the case of AF-3)
    binned_distances = torch.bucketize(distance_map, bins)
    crd_mask_rep_atom_II = crd_mask_rep_atoms_I.unsqueeze(
        -1
    ) * crd_mask_rep_atoms_I.unsqueeze(-2)
    distogram_cce = cce_loss(pred_distogram.permute(0, -1, -2, -3), binned_distances)
    w = crd_mask_rep_atom_II.to(distogram_cce.dtype)
    if pair_weight is not None:
        w = w * pair_weight
    return (distogram_cce * w).sum() / (w.sum() + 1e-4)


class DistogramLoss(nn.Module):
    def __init__(self, weight, alpha_hotspot_distogram=0.0):
        super().__init__()
        self.weight = weight
        self.alpha_hotspot_distogram = alpha_hotspot_distogram
        self.cce_loss = nn.CrossEntropyLoss(reduction="none")
        self.eps = 1e-4

    def forward(self, network_input, network_output, loss_input):
        pred_distogram = network_output["distogram"]
        X_rep_atoms_I = loss_input["coord_token_lvl"]
        crd_mask_rep_atoms_I = loss_input["mask_token_lvl"]

        pair_weight = None
        if self.alpha_hotspot_distogram > 0:
            # OR-aggregate atom-level hotspot mask onto tokens via atom_to_token_map.
            # Matches the aggregation used in metrics/interface_hits.py.
            atom_to_token = network_input["f"]["atom_to_token_map"]  # [B, L], -1 pad
            mask_atom = network_input["f"]["mask_hotspot_1_atom"]    # [B, L] bool
            B, I = pred_distogram.shape[:2]
            valid = (atom_to_token >= 0) & mask_atom.bool()
            # scatter_reduce_ with amax gives a boolean OR over atoms mapped to the same token.
            hot_token = torch.zeros(B, I, dtype=torch.uint8, device=pred_distogram.device)
            hot_token.scatter_reduce_(
                1,
                atom_to_token.clamp(min=0).long(),
                valid.to(torch.uint8),
                reduce="amax",
                include_self=True,
            )
            w_i = hot_token.unsqueeze(-1).to(pred_distogram.dtype)  # [B, I, 1]
            w_j = hot_token.unsqueeze(-2).to(pred_distogram.dtype)  # [B, 1, I]
            pair_weight = 1.0 + self.alpha_hotspot_distogram * torch.maximum(w_i, w_j)

        loss = distogram_loss(
            pred_distogram,
            X_rep_atoms_I,
            crd_mask_rep_atoms_I,
            self.cce_loss,
            pair_weight=pair_weight,
        )
        return self.weight * loss, {"distogram_loss": loss.detach()}


class MaskedTokenRecoveryLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, network_input, network_output, loss_input):
        pred_token_logits = network_output["masked_token_logits"].unsqueeze(0)  # (B, L, C)
        true_token_ids = loss_input["true_token_ids"].unsqueeze(0)  # (B, L,
        masked_tokens = loss_input["masked_tokens"]

        # No masked tokens → no loss to compute, but keep gradient path alive for DDP
        if masked_tokens.sum() == 0:
            zero = network_output["masked_token_logits"].sum() * 0.0
            return self.weight * zero, {"masked_token_recovery_loss": zero.detach()}

        #take only the masked tokens into account for loss
        pred_token_logits = pred_token_logits[:, masked_tokens, :]
        true_token_ids = true_token_ids[:, masked_tokens]
        # print('predictions', pred_token_logits.shape, pred_token_logits.argmax(dim=-1))
        # print('true', true_token_ids.shape, true_token_ids)

        token_loss = self.ce_loss(
            pred_token_logits.permute(0, 2, 1), true_token_ids
        )  # (B,L)
        token_loss = torch.mean(token_loss)
        return self.weight * token_loss, {"masked_token_recovery_loss": token_loss.detach()}


class NullLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, network_input, network_output, loss_input):
        loss = 0
        for key, val in network_output.items():
            val[val.isnan()] = 0
            loss += torch.sum(val) * 0

        return loss, {}
