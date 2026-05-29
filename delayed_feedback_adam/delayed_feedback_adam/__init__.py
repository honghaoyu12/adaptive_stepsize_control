"""Delayed-feedback Adam optimizer.

This package contains a PyTorch optimizer that wraps Adam-style directions with
an outer controller. The controller uses the previous step's observed loss
change to update a global learning-rate multiplier, avoiding the extra forward
pass required by a same-step actual-vs-predicted decrease controller.
"""

from .optimizer import DelayedFeedbackAdam

__all__ = ["DelayedFeedbackAdam"]
