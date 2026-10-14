from os import PathLike
from pathlib import Path
from dataclasses import asdict

from atomworks.ml.example_id import parse_example_id
from beartype.typing import Any
from rf3.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
    dump_structures,
    dump_trajectories,
)

from foundry.callbacks.callback import BaseCallback


class DumpValidationStructuresCallback(BaseCallback):
    """Dump predicted structures and/or diffusion trajectories during validation"""

    def __init__(
        self,
        save_dir: PathLike,
        dump_predictions: bool = False,
        one_model_per_file: bool = False,
        dump_trajectories: bool = False,
        compress_outputs: bool = False,
        annotate_b_factor_with_plddt: bool = False,
    ):
        """
        Args:
            dump_predictions: Whether to dump structures (CIF files) after validation batches.
            one_model_per_file: If True, write each structure within a diffusion batch to its own CIF files. If False,
                include each structure within a diffusion batch as a separate model within one CIF file.
            dump_trajectories: Whether to dump denoising trajectories after validation batches.
            compress_outputs: Whether to gzip output files. Defaults to ``False``.
            annotate_b_factor_with_plddt: Whether to write per-atom pLDDT into the B-factor
                column of dumped predictions. Defaults to ``False``.
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.dump_predictions = dump_predictions
        self.dump_trajectories = dump_trajectories
        self.one_model_per_file = one_model_per_file
        self.compress_outputs = compress_outputs
        self.annotate_b_factor_with_plddt = annotate_b_factor_with_plddt

    def on_validation_batch_end(
        self,
        trainer,
        outputs: dict,
        batch: Any,
        dataset_name: str | None,
        **_,
    ):
        if (not self.dump_predictions) and (not self.dump_trajectories):
            return  # Nothing to do

        assert (
            "network_output" in outputs
        ), "Validation outputs must contain `network_output` to dump structures!"

        network_output = outputs["network_output"]
        example = asdict(batch)  # Assume batch size = 1
        data = example["data_samples"]

        try:
            # ... try to extract the PDB ID and assembly ID from the example ID
            parsed_id = parse_example_id(data["example_id"][0])
            identifier = f"{parsed_id['pdb_id']}_{parsed_id['assembly_id']}"
        except (KeyError, ValueError, TypeError):
            # ... if parsing fails, fall back to the original example ID
            identifier = data["example_id"][0]

        def _build_path_from_example_id(dir: str, extra: str = "") -> Path:
            """Helper function to build a path from a training or validation example_id."""
            path = self.save_dir / dir / f"epoch_{trainer.state['current_epoch']}"

            path = path / dataset_name

            return path / f"{identifier}{extra}"

        # Determine file type based on compression setting
        file_type = "cif.gz" if self.compress_outputs else "cif"

        if self.dump_predictions:
            # network_output["X_L"] is already [D, L, 3] (diffusion samples, no separate batch
            # dim -- see af3_losses.py/distogram.py's own "[D, L, 3]" annotations), which is
            # exactly the (nbatch, n_atoms, 3) build_stack_from_atom_array_and_batched_coords
            # wants (it treats the diffusion-sample axis as "batch", one CIF model per sample).
            # Indexing [0] here stripped it to 2D and crashed.
            atom_array_stack = build_stack_from_atom_array_and_batched_coords(
                network_output["X_L"], data["atom_array"][0]
            )

            if "plddt" in network_output and self.annotate_b_factor_with_plddt:
                confidence_loss_cfg = self.trainer.state[
                    "train_cfg"
                ].trainer.loss.confidence_loss

                confidence_outs = compile_af3_confidence_outputs(
                    plddt_logits=network_output["plddt"],
                    pae_logits=network_output["pae"],
                    pde_logits=network_output["pde"],
                    chain_iid_token_lvl=data["ground_truth"]["chain_iid_token_lvl"],
                    is_real_atom=data["confidence_features"]["is_real_atom"],
                    example_id=data["example_id"],
                    confidence_loss_cfg=confidence_loss_cfg,
                )

                # Annotate the B-factors of the predicted structures with the pLDDT scores
                # (Forces one model per file, if `one_model_per_file` is False)
                atom_array_stack = annotate_atom_array_b_factor_with_plddt(
                    atom_array_stack,
                    confidence_outs["plddt"],
                    data["confidence_features"]["is_real_atom"],
                )
                self.one_model_per_file = True

            dump_structures(
                atom_arrays=atom_array_stack,
                base_path=_build_path_from_example_id("predictions"),
                one_model_per_file=self.one_model_per_file,
                file_type=file_type,
            )

        if self.dump_trajectories:
            # Each entry of X_*_L_traj has shape [B, D, L, 3]; take batch index 0.
            dump_trajectories(
                trajectory_list=[t[0] for t in network_output["X_denoised_L_traj"]],
                atom_array=data["atom_array"][0],
                base_path=_build_path_from_example_id("trajectories", "_denoised"),
                file_type=file_type,
            )
            dump_trajectories(
                trajectory_list=[t[0] for t in network_output["X_noisy_L_traj"]],
                atom_array=data["atom_array"][0],
                base_path=_build_path_from_example_id("trajectories", "_noisy"),
                file_type=file_type,
            )
