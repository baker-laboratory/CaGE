import torch
from beartype.typing import Any, Literal
from jaxtyping import Float

from foundry.utils.ddp import RankedLogger
from foundry.utils.rigid import rot_vec_mul
from foundry.utils.rotation_augmentation import centre_random_augmentation

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def map_trajectories_to_final_frame(
    frame_transforms: list[tuple[torch.Tensor, torch.Tensor]],
    trajectories: list[list[torch.Tensor | None]],
) -> None:
    """Map per-step trajectory frames, in place, into the frame of the final X_L.

    Every step re-centres and randomly rotates X_L, so step s is recorded in its own frame.
    `frame_transforms[s]` is step s's augmentation (R, t), X_L = R·X_L_prev + t; composing
    those of steps s+1..N maps step s onto the final coordinates exactly. None frames are skipped.
    """
    R_tot = t_tot = None  # identity: the last step is already in the final frame
    for step in reversed(range(len(frame_transforms))):
        if R_tot is not None:
            for traj in trajectories:
                if traj[step] is not None:
                    traj[step] = rot_vec_mul(R_tot[:, :, None], traj[step]) + t_tot
        R_s, t_s = frame_transforms[step]
        if R_tot is None:
            R_tot, t_tot = R_s, t_s
        else:
            R_tot, t_tot = R_tot @ R_s, rot_vec_mul(R_tot[:, :, None], t_s) + t_tot


class SampleDiffusion:
    """Algorithm 18"""

    # Subclasses that truncate the noise schedule set this False: the motif-guidance
    # window is defined on the full-schedule step fraction, which a truncated schedule
    # renormalizes onto a different noise range (see SamplePartialDiffusion).
    supports_motif_guidance: bool = True

    def __init__(
        self,
        *,
        # Hyperparameters
        num_timesteps: int,  # AF-3: 200
        min_t: int,  # AF-3: 0
        max_t: int,  # AF-3: 1
        sigma_data: int,  # AF-3: 16
        s_min: float,  # AF-3: 4e-4
        s_max: int,  # AF-3: 160
        p: int,  # AF-3: 7
        gamma_0: float,  # AF-3: 0.8
        gamma_min: float,  # AF-3: 1.0,
        noise_scale: float,  # AF-3: 1.003,
        step_scale: float,  # AF-3: 1.5,
        solver: Literal["af3"],
    ):
        """Initialize the diffusion sampler, to perform a complete diffusion roll-out with the given recycling outputs.

        We do not use default values for the parameters to make the Hydra configuration the single source of truth and avoid silent failures.

        Args:
            num_timesteps (int): The number of timesteps for which the noise schedule is constructed. Default is 200, per AF3.
            min_t (float): The minimum value of t in the schedule. Default is 0, per AF3.
            max_t (float): The maximum value of t in the schedule. Default is 1, per AF3.
            sigma_data (int): A constant determined by the variance of the data. Default is 16, as defined in the AlphaFold 3 Supplement (Algorithm 20, Diffusion Module).
            s_min (float): The minimum value of the noise schedule. Default is 4e-4, per AF3.
            s_max (float): The maximum value of the noise schedule. Default is 160, per AF3.
            p (int): A constant that determines the shape of the noise schedule. Default is 7, per AF3.
            gamma_0 (float): The value of gamma when t > gamma_min. Default is 0.8, per AF3.
            solver (str): The solver to use for the diffusion process. Default is "af3".

            TODO: Continue documentation of the remaining parameters.
        """
        self.num_timesteps = num_timesteps
        self.min_t = min_t
        self.max_t = max_t
        self.sigma_data = sigma_data
        self.s_min = s_min
        self.s_max = s_max
        self.p = p
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.solver = solver

    def _construct_inference_noise_schedule(self, device: torch.device) -> torch.Tensor:
        """Constructs a noise schedule for use during inference.

        The inference noise schedule is defined in the AF-3 supplement as:

            t_hat = sigma_data * (s_max**(1/p) + t * (s_min**(1/p) - s_max**(1/p)))**p

        Returns:
            torch.Tensor: A tensor representing the noise schedule `t_hat`.

        Reference:
            AlphaFold 3 Supplement, Section 3.7.1.
        """
        # Create a linearly spaced tensor of timesteps between min_t and max_t
        t = torch.linspace(self.min_t, self.max_t, self.num_timesteps, device=device)

        # Construct the noise schedule, using the formula provided in the reference
        t_hat = (
            self.sigma_data
            * (
                (self.s_max) ** (1 / self.p)
                + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )

        return t_hat

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        B: int,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
    ) -> torch.Tensor:
        """Sample initial point cloud from a normal distribution.

        Args:
            c0 (torch.Tensor): A scalar tensor that will be used to scale the initial point cloud. Effectively, the same as
                directly changing the standard deviation of the normal distribution. Derived from noise_schedule[0].
            B (int): The batch size from the pairformer
            D (int): The number of structures to sample.
            L (int): The number of atoms in the structure.
            coord_atom_lvl_to_be_noised (torch.Tensor): The atom-level coordinates to be noised (either completely or partially)
        """
        noise = c0 * torch.normal(
            mean=0.0, std=1.0, size=(B, D, L, 3), device=c0.device
        )
        # fd ** if running full trajectory do not inject inital coordinates
        X_L = noise  # + coord_atom_lvl_to_be_noised

        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        S_inputs_I: Float[torch.Tensor, "I c_s_inputs"],
        S_trunk_I: Float[torch.Tensor, "I c_s"],
        Z_trunk_II: Float[torch.Tensor, "I I c_z"],
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        is_padding_L: Float[torch.Tensor, "B L"],
        is_padding_I: Float[torch.Tensor, "B I"],
        is_padding_chiral: Float[torch.Tensor, "B num_chiral_centers"],
        # Classifier-free guidance (composable). The main stream (S_*/f above) is the
        # unconditional Δ∅. Each element of `cfg_guidance_streams` is a per-condition
        # (conditioned) stream: a dict {f, S_inputs_I, S_trunk_I, Z_trunk_II, scale}.
        # The guided direction is delta_∅ + Σ_j scale_j * (delta_j - delta_∅). Optional so
        # existing callers (e.g. the training mini-rollout) are unaffected.
        inference_classifier_free_guidance: bool = False,
        cfg_guidance_streams: list[dict[str, Any]] | None = None,
        # Guidance interval for classier-free guidance
        # [cfg_sigma_min, cfg_sigma_max]. Defaults (0, +inf)
        cfg_sigma_min: float = 0.0,
        cfg_sigma_max: float | None = None,
        # Motif guidance: a sampler-side constraint on x0, NOT a trained condition.
        # Duck-typed so this module keeps no dependency on rf3.guidance: anything with
        # `.refine(x0, t_norm=float)` works. None is an exact no-op.
        motif_guidance: Any = None,
    ) -> dict[str, Any]:
        """Perform a complete diffusion roll-out with the given recycling outputs.

        Args:
            diffusion_module (torch.nn.Module): The diffusion module to use for denoising. If using EMA and performing validation or inference,
                this model should be the EMA model.
        """
        # Construct the noise schedule t_hat for inference on the appropriate device
        noise_schedule = self._construct_inference_noise_schedule(
            device=S_inputs_I.device
        )
        # Euler step count, taken from the schedule rather than self.num_timesteps so a
        # truncating subclass cannot silently rescale the motif-guidance t_norm window.
        n_euler_steps = max(1, noise_schedule.numel() - 1)

        # Guidance noise-level interval; cfg_sigma_max=None means +inf.
        cfg_sigma_max = float("inf") if cfg_sigma_max is None else float(cfg_sigma_max)
        if cfg_sigma_min > cfg_sigma_max:
            raise ValueError(
                f"cfg_sigma_min ({cfg_sigma_min}) must not exceed cfg_sigma_max "
                f"({cfg_sigma_max})."
            )
        n_guided_steps = 0
        n_motif_steps = 0

        # Infer number of atoms from any atom-level feature
        B, L = f["ref_element"].shape[:2]
        D = diffusion_batch_size

        if motif_guidance is not None:
            if not self.supports_motif_guidance:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support motif guidance: its noise "
                    "schedule is truncated, so the step-fraction t_norm the guidance "
                    "window is defined on no longer matches the full schedule."
                )
            # The motif feats index one example's L axis; a real pairformer batch would
            # pull the wrong atoms.
            if B != 1:
                raise ValueError(f"Motif guidance requires B == 1, got B={B}")

        # Initial X_L is drawn from a normal distribution with a mean vector of 0 and a
        # covariance matrix equal to the 3x3 identity matrix, scaled by the noise schedule
        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            B=B,
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
        )  # (D, L, 3)

        use_cfg = inference_classifier_free_guidance and bool(cfg_guidance_streams)

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        t_hats = []
        # CFG only: the guided clean estimate, and each stream's own one (None on unguided steps)
        X_denoised_L_guided_traj = []
        X_denoised_L_stream_trajs = (
            [[] for _ in cfg_guidance_streams] if use_cfg else []
        )
        # Each step's rigid augmentation, to map the trajectories into the final frame
        frame_transforms = []

        for step_i, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            # (All predicted atoms exist)
            X_exists_L = torch.ones((B, D, L)).bool()  # (B, D, L)

            # Apply a random rotation and translation to the structure
            # TODO: Make s_trans a hyperparameter
            s_trans = 1.0
            X_L_prev = X_L
            X_L, R_aug = centre_random_augmentation(
                X_L, X_exists_L, s_trans, return_rotation=True
            )
            # Rigid, so one translation per sample: X_L = R·X_L_prev + t
            t_aug = (X_L - rot_vec_mul(R_aug[:, :, None], X_L_prev)).mean(
                dim=-2, keepdim=True
            )
            frame_transforms.append((R_aug.cpu(), t_aug.cpu()))

            # Update gamma
            gamma = self.gamma_0 if c_t > self.gamma_min else 0

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            X_noisy_L = X_L + epsilon_L

            # Denoise the coordinates
            X_denoised_L = diffusion_module(
                X_noisy_L=X_noisy_L,
                t=t_hat.expand(B, D),  # (B, D)
                f=f,
                S_inputs_I=S_inputs_I,
                S_trunk_I=S_trunk_I,
                Z_trunk_II=Z_trunk_II,
                is_padding_L=is_padding_L,
                is_padding_I=is_padding_I,
                is_padding_chiral=is_padding_chiral,
            )

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat.
            # In composable CFG the main stream above is the unconditional Δ∅.
            delta_L = (X_noisy_L - X_denoised_L) / t_hat
            # clone, not alias: the guided-x0 reconstruction below is (delta_L -
            # delta_uncond), so an in-place update to delta_L anywhere would collapse
            # that difference to zero and silently refine the unconditional estimate.
            delta_uncond = delta_L.clone()

            # Only guide on the requested interval;
            in_cfg_interval = cfg_sigma_min <= float(t_hat) <= cfg_sigma_max
            if use_cfg and in_cfg_interval:
                n_guided_steps += 1
                # Composable CFG: delta_∅ + Σ_j scale_j * (delta_j - delta_∅).
                for stream, stream_traj in zip(
                    cfg_guidance_streams, X_denoised_L_stream_trajs
                ):
                    X_denoised_L_j = diffusion_module(
                        X_noisy_L=X_noisy_L,
                        t=t_hat.expand(B, D),
                        f=stream["f"],
                        S_inputs_I=stream["S_inputs_I"],
                        S_trunk_I=stream["S_trunk_I"],
                        Z_trunk_II=stream["Z_trunk_II"],
                        is_padding_L=is_padding_L,
                        is_padding_I=is_padding_I,
                        # The stream's chirals feature was collated with the stream, so
                        # it needs the stream's own chiral padding, not the main one's.
                        is_padding_chiral=stream.get("is_padding_chiral")
                        if stream.get("is_padding_chiral") is not None
                        else is_padding_chiral,
                    )
                    stream_traj.append(X_denoised_L_j.cpu())
                    delta_L_j = (X_noisy_L - X_denoised_L_j) / t_hat
                    r_j = delta_L_j - delta_uncond
                    delta_L = delta_L + stream["scale"] * r_j
            else:
                for stream_traj in X_denoised_L_stream_trajs:
                    stream_traj.append(None)

            if use_cfg or motif_guidance is not None:
                # The clean estimate the (guided) update below moves toward. Taken relative
                # to the main one: x_t - t_hat * delta cancels ~1e-3 Å away at x_t ~ 1e4 Å
                x0_guided = X_denoised_L - t_hat * (delta_L - delta_uncond)

                # Motif guidance acts here, AFTER the CFG blend, so it refines the x0 the
                # Euler update actually moves toward rather than the unconditional one.
                if motif_guidance is not None:
                    t_norm = 1.0 - float(step_i) / float(n_euler_steps)
                    x0_refined = motif_guidance.refine(x0_guided, t_norm=t_norm)
                    # Identity, not torch.equal: refine returns its argument when the
                    # gate is shut or mu <= 0, so this is exact and costs no device sync.
                    if x0_refined is not x0_guided:
                        n_motif_steps += 1
                        # Correction ADDED to delta_L. Recomputing it as
                        # (X_noisy_L - x0_refined) / t_hat instead would throw away the
                        # CFG term accumulated above and reintroduce the cancellation.
                        delta_L = delta_L - (x0_refined - x0_guided) / t_hat
                        x0_guided = x0_refined

                X_denoised_L_guided_traj.append(x0_guided.cpu())

            d_t = c_t - t_hat

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + self.step_scale * d_t * delta_L

            X_noisy_L_scaled = (
                X_noisy_L
                / (torch.sqrt(t_hat[..., None, None] ** 2 + self.sigma_data**2))
            ) * self.sigma_data
            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_traj.append(X_noisy_L_scaled.cpu())
            X_denoised_L_traj.append(X_denoised_L.cpu())
            t_hats.append(t_hat)

        # Warn: motif guidance was attached but never applied. The engine logs only the
        # arming transition, so without this an inert run is indistinguishable from a
        # working one -- either the t_norm window excluded every step, or no sample's
        # anchor RMSD ever fell below the arm threshold.
        if motif_guidance is not None and n_motif_steps == 0:
            ranked_logger.warning(
                "Motif guidance was attached but never applied: no step fell inside the "
                "t_norm window, or no diffusion sample ever armed."
            )

        # Warn: no diffusion step within guidance interval
        if use_cfg and n_guided_steps == 0:
            ranked_logger.warning(
                "CFG requested but no diffusion step fell in the noise interval "
                f"[{cfg_sigma_min}, {cfg_sigma_max}]; guidance had no effect."
            )

        map_trajectories_to_final_frame(
            frame_transforms,
            [
                traj
                for traj in (
                    X_noisy_L_traj,
                    X_denoised_L_traj,
                    X_denoised_L_guided_traj,
                    *X_denoised_L_stream_trajs,
                )
                if traj
            ],
        )

        return dict(
            X_L=X_L,  # (D, L, 3)
            # Trajectories: one entry per step, all in the frame of the final X_L
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            # CFG only (empty otherwise)
            X_denoised_L_guided_traj=X_denoised_L_guided_traj,
            X_denoised_L_stream_trajs={
                stream.get("name") or f"condition_{idx}": traj
                for idx, (stream, traj) in enumerate(
                    zip(cfg_guidance_streams or [], X_denoised_L_stream_trajs)
                )
            },
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
        )


class SamplePartialDiffusion(SampleDiffusion):
    # This class only overrides the schedule and the initial structure; it inherits
    # sample_diffusion_like_af3, so a class flag is the only way the inherited body can
    # refuse motif guidance. Truncating the schedule renormalizes the step-fraction
    # t_norm onto the surviving tail, so the guidance window would silently shift.
    supports_motif_guidance = False

    def __init__(self, partial_t: int, **kwargs):
        super().__init__(**kwargs)
        self.partial_t = partial_t

    def _construct_inference_noise_schedule(self, device: torch.device) -> torch.Tensor:
        """Constructs a noise schedule for use during inference with partial t."""
        t_hat_full = super()._construct_inference_noise_schedule(device)

        assert (
            self.partial_t < self.num_timesteps
        ), f"Partial t ({self.partial_t}) must be less than num_timesteps ({self.num_timesteps})"
        ranked_logger.info(
            f"Using partial t index: {self.partial_t} [e.g., {t_hat_full[self.partial_t]:.4}], or {self.partial_t / (self.num_timesteps):.2%}, by index (100% is data, 0% is noise)"
        )

        return t_hat_full[self.partial_t :]

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        B: int,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
    ) -> torch.Tensor:
        noise = c0 * torch.normal(
            mean=0.0, std=1.0, size=(B, D, L, 3), device=c0.device
        )
        # partial diffusion: inject coords to be noised
        X_L = noise + coord_atom_lvl_to_be_noised

        return X_L
