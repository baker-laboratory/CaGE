from collections import defaultdict
from collections.abc import Callable
from typing import Literal
from functools import partial

import biotite.structure as struc
import numpy as np
from atomworks.common import exists
from atomworks.io.utils.atom_array import apply_and_spread
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.query import QueryExpression
from atomworks.io.utils.scatter import (
    apply_segment_wise,
    spread_segment_wise,
)
from atomworks.ml.utils.token import apply_and_spread_token_wise
from atomworks.ml.conditions import (
    C_CRD,
    C_DIS,
    C_IDX,
    C_SEQ,
)
from atomworks.ml.conditions.annotator import ensure_annotations
from atomworks.ml.transforms import mask_generator as mg
from atomworks.ml.transforms.crop import resize_crop_info_if_too_many_atoms
from atomworks.ml.transforms.design_task import DesignTaskABC
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from atomworks.ml.utils.token import get_token_count, get_token_starts
from biotite.structure import AtomArray
from foundry.common import true_with_prob
from foundry.utils.ddp import RankedLogger
from scipy.spatial import ConvexHull
from scipy.special import softmax
from rf3.data.design_tasks.design_task_modifiers import add_rasa_condition, add_hbond_condition

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

class UnconditionalTask(DesignTaskABC):
    """Sample a protein backbone & sequence unconditionally."""

    def can_apply(self, data: dict) -> bool:
        return True

    def forward(self, data: dict) -> dict:
        atom_array = as_atom_array_plus(data["atom_array"])
        n_atoms = atom_array.array_length()

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            # required fields
            "class": self.__class__.__name__,
            "mask": np.zeros(n_atoms, dtype=bool),
        }
        data["atom_array"] = atom_array
        return data


# DesignTask with DesignTaskModifiers

RFD3_DESIGN_TASK_DEFAULTS = {
    "rasa_kwargs": {
        "p_add_rasa_condition": 0.0,  # 0.6 in RFD3
        "probe_radius": 1.4,
        "atom_radii": "ProtOr",
        "point_number": 100,
        "p_include_non_ligand": 0.0,  # 0.1 in RFD3
        "rasa_max": 0.2,
    },
    "hbond_kwargs": {
        "p_add_hbond_condition": 0.0,  # 0.2 in RFD3
        "cutoff_HA_dist": 2.2,
        "cutoff_DA_distance": 3.2,
        "p_hbond_subsample": 1.0,  # 0.5 in RFD3
    },
}


class RFD3DesignTask(DesignTaskABC):
    """Wrapper class for DesignTasks used to train RFD3

    The concept behind this class is to allow for common functions to be called by each
    design task without having to specify all kwargs in each design task separately.
    All kwargs and default values are specified in the RFD3_DESIGN_TASK_DEFAULTS constant.

    **kwargs: See RFD3_DESIGN_TASK_DEFAULTS for details.
    """

    # Overrides for the RFD3_DESIGN_TASK_DEFAULTS. Subclasses can modify their defaults or set to {}
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: dict[str, dict]

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, "RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES"):
            raise TypeError(
                f"{cls.__name__} must define class variable " "`RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES` (even if empty)."
            )

    @staticmethod
    def _deep_merge_dicts(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries, with override values taking precedence.

        Args:
            base: The base dictionary
            override: The dictionary with override values

        Returns:
            A new dictionary with merged values
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = RFD3DesignTask._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def __init__(self, global_design_task_kwargs: dict | None = None, **kwargs):
        super().__init__()
        # Set default values for all design task modifiers with proper deep merge
        design_task_defaults = self._deep_merge_dicts(
            RFD3_DESIGN_TASK_DEFAULTS, self.RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES
        )

        # Handle any global args that were passed, with task-specific kwargs taking precedence
        if not exists(global_design_task_kwargs):
            global_design_task_kwargs = {}
        kwargs = self._deep_merge_dicts(global_design_task_kwargs, kwargs)

        for key, value in design_task_defaults.items():
            params_to_use = value
            if key in kwargs:
                # Deep merge with any provided kwargs
                params_to_use = self._deep_merge_dicts(params_to_use, kwargs[key])
            setattr(self, key, params_to_use)

    def apply_design_task_modifiers(self, data: dict, overrides: dict = {}) -> dict:
        # Make local copies to avoid mutating instance state
        rasa_kwargs = self.rasa_kwargs.copy() | overrides.get("rasa_kwargs", {})
        hbond_kwargs = self.hbond_kwargs.copy() | overrides.get("hbond_kwargs", {})

        # Set defaults for task dictionary
        data["task"] |= {
            "added_rasa_condition": False,
            "added_hbond_condition": False,
        }

        # Possibly add Conditions corresponding to design task modifiers
        if true_with_prob(rasa_kwargs.pop("p_add_rasa_condition")):
            data = add_rasa_condition(data, **rasa_kwargs)
            data["task"]["added_rasa_condition"] = True
        if true_with_prob(hbond_kwargs.pop("p_add_hbond_condition")):
            data = add_hbond_condition(data, **hbond_kwargs)
            # add_hbond_condition is a no-op when hbplus is unavailable, so take the flag
            # from what it actually did rather than from the coin flip.
            data["task"]["added_hbond_condition"] = not data.get("log_dict", {}).get(
                "hbond_calculation_failed", False
            )
        return data
