import torch
import torch.nn as nn
from rf3.model.layers.utils.layer_utils import linearNoBias, Transition, NormBiasConfig
from rf3.model.layers.embedders.relative_positional_encoding import (
    RelativePositionEncoding,
)
from rf3.model.layers.diffusion.decoder import AtomAttentionDecoder
from rf3.model.layers.diffusion.encoder import AtomAttentionEncoderDiffusion
from rf3.model.layers.diffusion.transformer import DiffusionTransformer

from foundry.model.layers.blocks import FourierEmbedding
from foundry.training.checkpoint import activation_checkpointing


class DiffusionModule(nn.Module):
    def __init__(
        self,
        sigma_data,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_z,
        f_pred,
        upcast_inputs,
        diffusion_conditioning,
        atom_attention_encoder,
        diffusion_transformer,
        atom_attention_decoder,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.f_pred = f_pred
        self.upcast_inputs = upcast_inputs

        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=sigma_data, c_s=c_s, c_z=c_z, norm_bias=norm_bias, **diffusion_conditioning
        )
        self.atom_attention_encoder = AtomAttentionEncoderDiffusion(
            c_token=c_token,
            c_s=c_s,
            c_atom=c_atom,
            c_atompair=c_atompair,
            norm_bias=norm_bias,
            **atom_attention_encoder,
        )
        self.process_s = nn.Sequential(
            nn.LayerNorm((c_s,), bias=norm_bias.projection),
            linearNoBias(c_s, c_token),
        )
        self.diffusion_transformer = DiffusionTransformer(
            c_token=c_token, c_s=c_s, c_tokenpair=c_z, norm_bias=norm_bias, **diffusion_transformer
        )
        self.layer_norm_1 = nn.LayerNorm(c_token, bias=norm_bias.projection)
        self.atom_attention_decoder = AtomAttentionDecoder(
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            norm_bias=norm_bias,
            **atom_attention_decoder,
        )

    def forward(
        self,
        X_noisy_L,  # [B, L, 3]
        t,  # [B] (0 is ground truth)
        f,  # Dict (Input feature dictionary)
        S_inputs_I,  # [B, I, C_S_input]
        S_trunk_I,  # [B, I, C_S_trunk]
        Z_trunk_II,  # [B, I, I, C_Z]
        is_padding_I,
        is_padding_L,
        is_padding_chiral,
    ):
        if self.upcast_inputs:
            S_inputs_I = S_inputs_I.float()
            S_trunk_I = S_trunk_I.float()
            Z_trunk_II = Z_trunk_II.float()
            X_noisy_L = X_noisy_L.float()
            f = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                 for k, v in f.items()}

        # Conditioning
        S_I, Z_II = self.diffusion_conditioning(t, f, S_inputs_I, S_trunk_I, Z_trunk_II)

        # Scale positions to dimensionless vectors with approximately unit variance
        if self.f_pred == "edm":
            R_noisy_L = X_noisy_L / torch.sqrt(
                t[..., None, None] ** 2 + self.sigma_data**2
            )
        elif self.f_pred == "unconditioned":
            R_noisy_L = torch.zeros_like(X_noisy_L)
        elif self.f_pred == "noise_pred":
            R_noisy_L = X_noisy_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        A_I, Q_skip_L, C_skip_L, P_skip_LL = self.atom_attention_encoder(
            f, R_noisy_L, S_trunk_I, Z_II, is_padding_L, is_padding_chiral
        )
        # Full self-attention on token level
        A_I = A_I + self.process_s(S_I)
        A_I = self.diffusion_transformer(A_I, S_I, Z_II, is_padding_I)
        A_I = self.layer_norm_1(A_I)

        if self.upcast_inputs:
            A_I = A_I.float()

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        R_update_L = self.atom_attention_decoder(
            f, A_I, Q_skip_L, C_skip_L, P_skip_LL, is_padding_L
        )
        # Rescale updates to positions and combine with input positions
        if self.f_pred == "edm":
            X_out_L = (self.sigma_data**2 / (self.sigma_data**2 + t**2))[
                ..., None, None
            ] * X_noisy_L + (self.sigma_data * t / (self.sigma_data**2 + t**2) ** 0.5)[
                ..., None, None
            ] * R_update_L
        elif self.f_pred == "unconditioned":
            X_out_L = R_update_L
        elif self.f_pred == "noise_pred":
            X_out_L = X_noisy_L + R_update_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")

        return X_out_L


class DiffusionConditioning(nn.Module):
    def __init__(
        self, sigma_data, c_z, c_s, c_s_inputs, c_t_embed, relative_position_encoding,
        norm_bias: NormBiasConfig = NormBiasConfig(),
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.relative_position_encoding = RelativePositionEncoding(
            c_z=c_z, **relative_position_encoding
        )
        self.to_zii = nn.Sequential(
            nn.LayerNorm(
                c_z * 2, bias=norm_bias.projection
            ),  # Operates on concatenated (z_ij_trunk: [..., c_z]), RelativePositionalEncoding: [..., c_z])
            linearNoBias(c_z * 2, c_z),
        )
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_z, n=2, norm_bias=norm_bias),
                Transition(c=c_z, n=2, norm_bias=norm_bias),
            ]
        )
        self.to_si = nn.Sequential(
            nn.LayerNorm(c_s + c_s_inputs, bias=norm_bias.projection), linearNoBias(c_s + c_s_inputs, c_s)
        )
        c_t_embed = 256
        self.fourier_embedding = FourierEmbedding(c_t_embed)
        self.process_n = nn.Sequential(
            nn.LayerNorm(c_t_embed, bias=norm_bias.projection), linearNoBias(c_t_embed, c_s)
        )
        self.transition_2 = nn.ModuleList(
            [
                Transition(c=c_s, n=2, norm_bias=norm_bias),
                Transition(c=c_s, n=2, norm_bias=norm_bias),
            ]
        )

    def forward(self, t, f, S_inputs_I, S_trunk_I, Z_trunk_II):
        # Pair conditioning
        Z_II = torch.cat([Z_trunk_II, self.relative_position_encoding(f)], dim=-1)

        @activation_checkpointing
        def _run_conditioning(Z_II, S_trunk_I, S_inputs_I):
            Z_II = self.to_zii(Z_II)
            for b in range(2):
                Z_II = Z_II + self.transition_1[b](Z_II)

            # Single conditioning
            S_I = torch.cat([S_trunk_I, S_inputs_I], dim=-1)
            S_I = self.to_si(S_I)
            N_D = self.fourier_embedding(1 / 4 * torch.log(t / self.sigma_data))
            S_I = self.process_n(N_D).unsqueeze(-2) + S_I[:, None]
            for b in range(2):
                S_I = S_I + self.transition_2[b](S_I)

            return S_I, Z_II

        return _run_conditioning(Z_II, S_trunk_I, S_inputs_I)


pi = torch.acos(torch.zeros(1)).item() * 2


class FourierEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.register_buffer("w", torch.zeros(c, dtype=torch.float32))
        self.register_buffer("b", torch.zeros(c, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # super().reset_parameters()
        nn.init.normal_(self.w)
        nn.init.normal_(self.b)

    def forward(
        self,
        t,  # [D]
    ):
        return torch.cos(2 * pi * (t[:, :, None] * self.w + self.b))
