# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from contextlib import contextmanager

import torch
import torch.nn as nn


class PlannerIdentifiedInputScale(nn.Module):
    """Positive input scale with explicit gradient ownership.

    The parameter belongs to the predictor so the existing optimizer, DDP
    wrapper, and checkpoint path manage it automatically. Its gradient is
    detached by default: the ordinary transition loss cannot identify it.
    A planner-landscape objective must opt in via ``landscape_gradient``.
    """

    def __init__(self, init_scale=1.0):
        super().__init__()
        if not math.isfinite(init_scale) or init_scale <= 0.0:
            raise ValueError(f"init_scale must be finite and positive, got {init_scale}")
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))
        self._landscape_gradient_enabled = False
        self._runtime_override = None

    @property
    def input_scale(self):
        return self.log_scale.exp()

    @property
    def runtime_override(self):
        return self._runtime_override

    @contextmanager
    def landscape_gradient(self):
        previous = self._landscape_gradient_enabled
        self._landscape_gradient_enabled = True
        try:
            yield self
        finally:
            self._landscape_gradient_enabled = previous

    def set_runtime_override(self, scale):
        if scale is None:
            self._runtime_override = None
            return
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"runtime scale must be finite and positive, got {scale}")
        self._runtime_override = scale

    def clear_runtime_override(self):
        self._runtime_override = None

    def effective_scale(self, reference):
        if self._runtime_override is not None:
            return reference.new_tensor(self._runtime_override)
        scale = self.input_scale.to(device=reference.device, dtype=reference.dtype)
        if not self._landscape_gradient_enabled:
            scale = scale.detach()
        return scale

    def forward(self, x):
        return x * self.effective_scale(x)


def configure_planner_identified_scale(module, mode="learned", value=None, checkpoint_log_scale=None):
    """Apply a runtime-only scale intervention and return an auditable record.

    Exactly one :class:`PlannerIdentifiedInputScale` must be present.  The
    learned parameter is never mutated: fixed interventions are stored in the
    module's non-checkpointed runtime override.
    """

    scales = [child for child in module.modules() if isinstance(child, PlannerIdentifiedInputScale)]
    if len(scales) != 1:
        raise RuntimeError(f"Expected exactly one planner-identified input scale, found {len(scales)}")
    scale = scales[0]
    learned_log_scale = float(scale.log_scale.detach().cpu())
    learned_scale = math.exp(learned_log_scale)
    if checkpoint_log_scale is not None and not math.isclose(
        learned_log_scale,
        float(checkpoint_log_scale),
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ):
        raise RuntimeError(
            "Loaded planner scale does not match the checkpoint: "
            f"module={learned_log_scale:.9g}, checkpoint={float(checkpoint_log_scale):.9g}"
        )

    mode = str(mode).lower()
    if mode == "learned":
        if value is not None:
            raise ValueError("A learned-scale intervention must not provide a fixed value")
        scale.clear_runtime_override()
        effective_scale = learned_scale
    elif mode == "fixed":
        if value is None:
            raise ValueError("A fixed-scale intervention requires a value")
        scale.set_runtime_override(value)
        effective_scale = float(value)
    else:
        raise ValueError(f"Unknown planner scale intervention mode: {mode!r}")

    return {
        "mode": mode,
        "requested_value": None if value is None else float(value),
        "checkpoint_log_scale": learned_log_scale,
        "checkpoint_learned_scale": learned_scale,
        "effective_scale": effective_scale,
        "parameter_unchanged": math.isclose(
            float(scale.log_scale.detach().cpu()), learned_log_scale, rel_tol=0.0, abs_tol=0.0
        ),
    }
