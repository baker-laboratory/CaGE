import torch
from beartype.typing import Any
from jaxtyping import Bool, Float

from foundry.metrics.metric import Metric
    

class MaskedTokenRecovery(Metric):
    """
    Metric to compute the masked token recovery accuracy.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "pred_tokens": ("network_output", "masked_token_logits"),
            "true_token_ids": ("extra_info", "true_token_ids"),
            "masked_tokens": ("extra_info", "masked_tokens"),
        }
    
    def compute(
        self,
        pred_tokens: Float[torch.Tensor, "I n_tokens"],
        true_token_ids: torch.Tensor,
        masked_tokens: Bool[torch.Tensor, "I"],
    ) -> dict[str, Any]:
        """
        Computes the masked token recovery accuracy.

        Args:
            pred_tokens: The predicted token logits. Shape: [I, n_tokens].
            true_tokens: The ground-truth tokens. Shape: [I].
            masked_tokens: A boolean mask indicating which tokens were masked. Shape: [I].
        """
        # Get the predicted tokens by taking the argmax over the logits
        pred_token_indices = torch.argmax(pred_tokens, dim=-1)  # Shape: [I]

        # Select only the masked tokens
        masked_true_tokens = true_token_ids[masked_tokens]  # Shape: [N_masked]
        masked_pred_tokens = pred_token_indices[masked_tokens]  # Shape: [N_masked]

        # Compute accuracy
        correct_predictions = (masked_true_tokens == masked_pred_tokens).sum().item()
        total_masked = masked_true_tokens.size(0)
        accuracy = correct_predictions / total_masked if total_masked > 0 else float("nan")

        return {"masked_token_recovery_accuracy": accuracy}