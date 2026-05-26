"""PI-controlled Muon optimizer wrapper for PyTorch.

This module implements an educational/research optimizer that combines:

1. Muon-style directions for matrix-valued parameters:
   SGD momentum followed by Newton-Schulz approximate orthogonalization.
2. AdamW-style directions for non-matrix parameters.
3. An outer-loop PI controller that adapts a global step multiplier alpha
   from the actual-vs-predicted decrease ratio on the same minibatch.

The intended use is experimental. It is not a drop-in replacement for highly
optimized production Muon implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


Closure = Callable[..., Tensor]


@dataclass
class PIControlStats:
    """Diagnostics from the most recent optimizer step."""

    loss_before: float
    loss_after: Optional[float]
    actual_decrease: Optional[float]
    predicted_decrease: Optional[float]
    rho: Optional[float]
    rho_bar: Optional[float]
    error: Optional[float]
    integral: float
    alpha: float
    log_alpha: float
    log_multiplier: float
    accepted: bool
    used_fallback_direction: bool
    grad_dot_direction: float
    backtracks: int = 0
    alpha_next: Optional[float] = None
    alpha_update_factor: Optional[float] = None
    trust_region_expanded: bool = False
    skipped_reason: Optional[str] = None


def _call_closure(closure: Closure, backward: bool) -> Tensor:
    """Call a closure that supports closure(backward=True/False).

    We require this convention because the PI controller needs one backward
    pass at the original parameters and one extra forward-only loss evaluation
    at the trial parameters on the same minibatch.
    """
    try:
        return closure(backward=backward)
    except TypeError as exc:
        raise TypeError(
            "PI-controlled optimizers require a closure with signature "
            "closure(backward: bool). The closure should compute the loss on "
            "the same minibatch; when backward=True it should also call "
            "loss.backward(), and when backward=False it should only return "
            "the forward loss."
        ) from exc


def newton_schulz_orthogonalize(
    matrix: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    use_bfloat16: bool = False,
) -> Tensor:
    """Approximate the polar/orthogonal factor of a 2D matrix.

    This is the Newton-Schulz-style quintic iteration commonly used in Muon.
    Given a matrix G, it approximately returns a semi-orthogonal matrix with
    the same singular vectors as G and singular values pushed toward 1.

    Parameters
    ----------
    matrix:
        A 2D tensor.
    steps:
        Number of Newton-Schulz iterations.
    eps:
        Numerical stabilizer for initial normalization.
    coefficients:
        Quintic coefficients (a, b, c) for
        X <- aX + b(XX^T)X + c(XX^T)^2X.
    use_bfloat16:
        If True and supported, run the iteration in bfloat16. For a small,
        portable research implementation, False is safer on CPU.
    """
    if matrix.ndim != 2:
        raise ValueError("newton_schulz_orthogonalize expects a 2D tensor.")

    original_dtype = matrix.dtype
    X = matrix
    if use_bfloat16 and matrix.device.type != "cpu":
        X = X.to(torch.bfloat16)
    else:
        # Keep at least float32 for numerical stability on CPU / small demos.
        if X.dtype in (torch.float16, torch.bfloat16):
            X = X.float()

    X = X / (X.norm() + eps)

    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    a, b, c = coefficients
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(original_dtype)


def _as_matrix(t: Tensor) -> Tuple[Tensor, Callable[[Tensor], Tensor]]:
    """Flatten a parameter/update tensor into a matrix for Muon.

    PyTorch's official ``torch.optim.Muon`` only supports 2D parameters.

    Returns the matrix view and a function that maps a matrix back to the
    original tensor shape.
    """
    original_shape = t.shape
    if t.ndim == 2:
        return t, lambda m: m.reshape(original_shape)
    raise ValueError("Muon matrix conversion supports only 2D tensors.")


def default_muon_param_groups(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    adamw_name_keywords: Sequence[str] = (
        "embed",
        "embedding",
        "lm_head",
        "head",
        "norm",
        "bias",
    ),
) -> List[Dict[str, Any]]:
    """Split parameters into Muon and AdamW fallback groups.

    This follows PyTorch's ``torch.optim.Muon`` convention: use Muon for 2D
    hidden matrix weights, and AdamW for scalar/vector parameters, convolution
    kernels, embeddings, heads, norms, and biases. The keyword filter is
    deliberately conservative.
    """
    muon_params: List[torch.nn.Parameter] = []
    adamw_params: List[torch.nn.Parameter] = []

    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        lname = name.lower()
        excluded_by_name = any(k in lname for k in adamw_name_keywords)
        is_matrix_like = p.ndim == 2
        if is_matrix_like and not excluded_by_name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    groups: List[Dict[str, Any]] = []
    if muon_params:
        groups.append({"params": muon_params, "use_muon": True})
    if adamw_params:
        groups.append({"params": adamw_params, "use_muon": False})
    return groups


class PIMuon(Optimizer):
    """Muon/AdamW optimizer with an outer-loop PI step-size controller.

    The optimizer builds a proposed direction p_t from Muon for matrix-like
    parameters and AdamW for other parameters, then applies

        theta_trial = theta + alpha_t p_t.

    It evaluates the same minibatch loss before and after the trial step and
    computes

        rho_t = (loss_before - loss_after) / (-alpha_t g_t^T p_t + eps).

    The PI controller adapts alpha_t:

        rho_bar_t = beta_rho rho_bar_{t-1} + (1 - beta_rho) rho_t
        e_t       = rho_bar_t - rho_star
        I_t       = lambda_i I_{t-1} + e_t
        log alpha <- log alpha + Kp e_t + Ki I_t.

    Parameters are updated in-place by the trial step. If reject_bad_steps is
    True and rho_t <= rho_min, the trial parameter change is rolled back, but
    optimizer states are not rolled back. For stochastic neural network
    training, reject_bad_steps=False is usually safer.
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        alpha0: float = 1e-3,
        rho_star: float = 0.8,
        kp: float = 0.05,
        ki: float = 0.001,
        beta_rho: float = 0.9,
        use_rho_ema: bool = True,
        integral_decay: float = 0.95,
        integral_clip: float = 5.0,
        alpha_min: float = 1e-7,
        alpha_max: float = 1e-1,
        multiplier_min: float = 0.8,
        multiplier_max: float = 1.25,
        rho_clip: Optional[float] = 10.0,
        predicted_eps: float = 1e-12,
        reject_bad_steps: bool = False,
        rho_min: float = 0.0,
        fallback_to_sgd_if_not_descent: bool = True,
        non_descent_shrink: float = 0.5,
        max_backtracks: int = 4,
        backtrack_shrink: float = 0.5,
        trust_region_expand: bool = True,
        trust_region_rho_threshold: float = 0.9,
        trust_region_alpha_threshold: float = 1e-4,
        trust_region_expand_factor: float = 1.5,
        # Muon hyperparameters
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        ns_use_bfloat16: bool = False,
        muon_shape_scale: bool = True,
        # AdamW fallback hyperparameters
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if alpha0 <= 0:
            raise ValueError("alpha0 must be positive.")
        if alpha_min <= 0 or alpha_max <= 0 or alpha_min > alpha_max:
            raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max.")
        if not (0 <= beta_rho < 1):
            raise ValueError("beta_rho must lie in [0, 1).")
        if not (0 <= integral_decay <= 1):
            raise ValueError("integral_decay must lie in [0, 1].")
        if multiplier_min <= 0 or multiplier_max <= 0 or multiplier_min > multiplier_max:
            raise ValueError("multipliers must satisfy 0 < min <= max.")
        if rho_clip is not None and rho_clip <= 0:
            raise ValueError("rho_clip must be positive when provided.")
        if predicted_eps <= 0:
            raise ValueError("predicted_eps must be positive.")
        if not (0.0 < non_descent_shrink < 1.0):
            raise ValueError("non_descent_shrink must be in (0, 1).")
        if max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative.")
        if not (0.0 < backtrack_shrink < 1.0):
            raise ValueError("backtrack_shrink must be in (0, 1).")
        if trust_region_rho_threshold < 0.0:
            raise ValueError("trust_region_rho_threshold must be non-negative.")
        if trust_region_alpha_threshold <= 0.0:
            raise ValueError("trust_region_alpha_threshold must be positive.")
        if trust_region_expand_factor <= 1.0:
            raise ValueError("trust_region_expand_factor must be > 1.")
        if not (0 <= muon_momentum < 1):
            raise ValueError("muon_momentum must lie in [0, 1).")
        beta1, beta2 = adam_betas
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("adam_betas must lie in [0, 1).")

        defaults = dict(
            use_muon=True,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        clipped_alpha0 = min(max(float(alpha0), float(alpha_min)), float(alpha_max))
        self.alpha = clipped_alpha0
        self.log_alpha = math.log(clipped_alpha0)
        self.rho_star = float(rho_star)
        self.kp = float(kp)
        self.ki = float(ki)
        self.beta_rho = float(beta_rho)
        self.use_rho_ema = bool(use_rho_ema)
        self.integral_decay = float(integral_decay)
        self.integral_clip = float(integral_clip)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.multiplier_min = float(multiplier_min)
        self.multiplier_max = float(multiplier_max)
        self.rho_clip = rho_clip
        self.predicted_eps = float(predicted_eps)
        self.reject_bad_steps = bool(reject_bad_steps)
        self.rho_min = float(rho_min)
        self.fallback_to_sgd_if_not_descent = bool(fallback_to_sgd_if_not_descent)
        self.non_descent_shrink = float(non_descent_shrink)
        self.max_backtracks = int(max_backtracks)
        self.backtrack_shrink = float(backtrack_shrink)
        self.trust_region_expand = bool(trust_region_expand)
        self.trust_region_rho_threshold = float(trust_region_rho_threshold)
        self.trust_region_alpha_threshold = float(trust_region_alpha_threshold)
        self.trust_region_expand_factor = float(trust_region_expand_factor)

        self.muon_momentum = float(muon_momentum)
        self.muon_nesterov = bool(muon_nesterov)
        self.ns_steps = int(ns_steps)
        self.ns_eps = float(ns_eps)
        self.ns_use_bfloat16 = bool(ns_use_bfloat16)
        self.muon_shape_scale = bool(muon_shape_scale)

        self.adam_betas = adam_betas
        self.adam_eps = float(adam_eps)

        self.rho_bar: Optional[float] = None
        self.integral = 0.0
        self.last_stats: Optional[PIControlStats] = None

    @torch.no_grad()
    def _build_muon_direction(self, p: Tensor, grad: Tensor, state: Dict[str, Any]) -> Tensor:
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        buf = state["momentum_buffer"]

        # Match torch.optim.Muon's source behavior:
        # buf <- lerp(buf, grad, 1 - momentum).
        buf.lerp_(grad, 1.0 - self.muon_momentum)

        if self.muon_nesterov:
            update_raw = grad.lerp(buf, self.muon_momentum)
        else:
            update_raw = buf

        mat, unflatten = _as_matrix(update_raw)
        ortho = newton_schulz_orthogonalize(
            mat,
            steps=self.ns_steps,
            eps=self.ns_eps,
            use_bfloat16=self.ns_use_bfloat16,
        )

        if self.muon_shape_scale:
            rows, cols = mat.shape
            # Common Muon-style shape scaling to avoid very small updates for
            # rectangular matrices. This is a practical convention rather than
            # part of the PI control idea itself.
            ortho = ortho * math.sqrt(max(1.0, rows / max(cols, 1)))

        # p_t is the direction added to parameters. Since ortho approximates
        # the gradient-like update to subtract, the direction is -ortho.
        return -unflatten(ortho)

    @torch.no_grad()
    def _build_adamw_direction(self, p: Tensor, grad: Tensor, state: Dict[str, Any]) -> Tensor:
        beta1, beta2 = self.adam_betas
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        step = state["step"]
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(self.adam_eps)
        adam_update = exp_avg.div(bias_correction1).div(denom)

        return -adam_update

    @torch.no_grad()
    def _build_directions(self) -> Tuple[List[Tuple[Tensor, Tensor]], float]:
        """Build directions and return [(param, direction), ...], g^T p."""
        directions: List[Tuple[Tensor, Tensor]] = []
        grad_dot_direction = 0.0

        for group in self.param_groups:
            use_muon = bool(group.get("use_muon", True))
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if torch.is_complex(p) or torch.is_complex(grad):
                    raise RuntimeError("PIMuon does not support complex parameters or gradients.")
                if grad.is_sparse:
                    raise RuntimeError("PIMuon does not support sparse gradients.")

                state = self.state[p]
                if use_muon and p.ndim == 2:
                    direction = self._build_muon_direction(p, grad, state)
                else:
                    direction = self._build_adamw_direction(p, grad, state)

                directions.append((p, direction))
                grad_dot_direction += float(torch.sum(grad * direction).item())

        return directions, grad_dot_direction

    @torch.no_grad()
    def _build_sgd_fallback_directions(self) -> Tuple[List[Tuple[Tensor, Tensor]], float]:
        directions: List[Tuple[Tensor, Tensor]] = []
        grad_dot_direction = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                direction = -grad
                directions.append((p, direction))
                grad_dot_direction += float(torch.sum(grad * direction).item())
        return directions, grad_dot_direction

    @torch.no_grad()
    def _apply_directions(self, directions: List[Tuple[Tensor, Tensor]], alpha: float) -> None:
        for p, direction in directions:
            weight_decay = self._weight_decay_for_param(p)
            if weight_decay != 0.0:
                p.mul_(1.0 - alpha * weight_decay)
            p.add_(direction, alpha=alpha)

    def _weight_decay_for_param(self, param: Tensor) -> float:
        for group in self.param_groups:
            if any(param is group_param for group_param in group["params"]):
                return float(group.get("weight_decay", 0.0))
        return 0.0

    @torch.no_grad()
    def _all_params(self) -> List[Tensor]:
        return [p for group in self.param_groups for p in group["params"] if p.requires_grad]

    @torch.no_grad()
    def _clone_params(self) -> List[Tensor]:
        return [p.detach().clone() for p in self._all_params()]

    @torch.no_grad()
    def _restore_params(self, originals: List[Tensor]) -> None:
        for p, original in zip(self._all_params(), originals):
            p.copy_(original)

    @torch.no_grad()
    def _set_trial_params(
        self,
        originals: List[Tensor],
        directions: List[Tuple[Tensor, Tensor]],
        alpha: float,
    ) -> None:
        for original, (p, direction) in zip(originals, directions):
            weight_decay = self._weight_decay_for_param(p)
            trial = original
            if weight_decay != 0.0:
                trial = trial * (1.0 - alpha * weight_decay)
            p.copy_(trial + alpha * direction)

    @torch.no_grad()
    def _clone_direction_params(self, directions: List[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        return [p.detach().clone() for p, _direction in directions]

    @torch.no_grad()
    def _restore_direction_params(
        self,
        originals: List[Tensor],
        directions: List[Tuple[Tensor, Tensor]],
    ) -> None:
        for original, (p, _direction) in zip(originals, directions):
            p.copy_(original)

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _shrink_after_non_descent(self) -> Tuple[float, float]:
        alpha_before = self.alpha
        alpha_next = min(max(alpha_before * self.non_descent_shrink, self.alpha_min), self.alpha_max)
        self.alpha = alpha_next
        self.log_alpha = math.log(alpha_next)
        return alpha_next, math.log(alpha_next / alpha_before)

    @torch.no_grad()
    def _update_controller(
        self,
        rho: float,
        *,
        alpha_used: float,
        backtracks: int,
    ) -> Tuple[float, float, float, float, float, bool]:
        if self.rho_clip is not None:
            rho = float(max(-self.rho_clip, min(self.rho_clip, rho)))

        if self.use_rho_ema:
            if self.rho_bar is None:
                self.rho_bar = rho
            else:
                self.rho_bar = self.beta_rho * self.rho_bar + (1.0 - self.beta_rho) * rho
        else:
            self.rho_bar = rho

        error = self.rho_bar - self.rho_star

        self.integral = self.integral_decay * self.integral + error
        if self.integral_clip is not None:
            self.integral = max(-self.integral_clip, min(self.integral_clip, self.integral))

        log_multiplier = self.kp * error + self.ki * self.integral
        log_multiplier = max(math.log(self.multiplier_min), min(math.log(self.multiplier_max), log_multiplier))

        trust_region_expanded = (
            self.trust_region_expand
            and backtracks == 0
            and self.rho_bar >= self.trust_region_rho_threshold
            and alpha_used <= self.trust_region_alpha_threshold
        )
        if trust_region_expanded:
            log_multiplier = max(log_multiplier, math.log(self.trust_region_expand_factor))

        self.log_alpha = math.log(alpha_used) + log_multiplier
        self.log_alpha = max(math.log(self.alpha_min), min(math.log(self.alpha_max), self.log_alpha))
        self.alpha = math.exp(self.log_alpha)

        return self.rho_bar, error, self.integral, log_multiplier, self.alpha, trust_region_expanded

    def step(self, closure: Closure) -> Tensor:  # type: ignore[override]
        """Perform one PI-controlled Muon step.

        The closure must use the same minibatch for both calls:

            def closure(backward: bool = True):
                optimizer.zero_grad()
                loss = loss_fn(model(x_batch), y_batch)
                if backward:
                    loss.backward()
                return loss
        """
        # First call: compute original loss and gradients.
        with torch.enable_grad():
            loss_before = _call_closure(closure, backward=True)

        loss_before_value = float(loss_before.detach().item())
        alpha_before = self.alpha

        with torch.no_grad():
            directions, grad_dot_direction = self._build_directions()
            used_fallback = False

            if not directions:
                self.last_stats = PIControlStats(
                    loss_before=loss_before_value,
                    loss_after=None,
                    actual_decrease=None,
                    predicted_decrease=None,
                    rho=None,
                    rho_bar=self.rho_bar,
                    error=None,
                    integral=self.integral,
                    alpha=alpha_before,
                    log_alpha=self.log_alpha,
                    log_multiplier=0.0,
                    accepted=False,
                    used_fallback_direction=False,
                    grad_dot_direction=0.0,
                    alpha_next=alpha_before,
                    alpha_update_factor=1.0,
                    skipped_reason="no gradients available",
                )
                return loss_before.detach()

            predicted_decrease = -alpha_before * grad_dot_direction
            if (
                self.fallback_to_sgd_if_not_descent
                and predicted_decrease <= self.predicted_eps
            ):
                directions, grad_dot_direction = self._build_sgd_fallback_directions()
                predicted_decrease = -alpha_before * grad_dot_direction
                used_fallback = True

            if not directions or predicted_decrease <= self.predicted_eps:
                alpha_next, log_multiplier = self._shrink_after_non_descent()
                self.last_stats = PIControlStats(
                    loss_before=loss_before_value,
                    loss_after=None,
                    actual_decrease=None,
                    predicted_decrease=None,
                    rho=None,
                    rho_bar=self.rho_bar,
                    error=None,
                    integral=self.integral,
                    alpha=alpha_before,
                    log_alpha=self.log_alpha,
                    log_multiplier=log_multiplier,
                    accepted=False,
                    used_fallback_direction=used_fallback,
                    grad_dot_direction=grad_dot_direction,
                    alpha_next=alpha_next,
                    alpha_update_factor=alpha_next / alpha_before,
                    skipped_reason="non-descent direction and fallback unavailable",
                )
                return loss_before.detach()

            original_params = self._clone_direction_params(directions)

        loss_after_value = loss_before_value
        predicted_decrease = 0.0
        actual_decrease = 0.0
        rho = float("nan")
        accepted = False
        alpha_used = alpha_before
        backtracks_taken = 0

        for backtracks in range(self.max_backtracks + 1):
            trial_alpha = alpha_before * (self.backtrack_shrink**backtracks)
            with torch.no_grad():
                self._set_trial_params(original_params, directions, trial_alpha)

            # Second call: same minibatch, forward-only loss at trial parameters.
            with torch.no_grad():
                loss_after = _call_closure(closure, backward=False)
            loss_after_value = float(loss_after.detach().item())

            predicted_decrease = -trial_alpha * grad_dot_direction
            actual_decrease = loss_before_value - loss_after_value
            rho = actual_decrease / (predicted_decrease + self.predicted_eps)
            alpha_used = trial_alpha
            backtracks_taken = backtracks

            if (not self.reject_bad_steps) or (rho > self.rho_min):
                accepted = True
                break

        if not accepted:
            with torch.no_grad():
                self._restore_direction_params(original_params, directions)

        rho_bar, error, integral, log_multiplier, new_alpha, trust_region_expanded = self._update_controller(
            rho if math.isfinite(rho) else self.rho_star - 1.0,
            alpha_used=alpha_used,
            backtracks=backtracks_taken if accepted else self.max_backtracks + 1,
        )

        self.last_stats = PIControlStats(
            loss_before=loss_before_value,
            loss_after=loss_after_value,
            actual_decrease=actual_decrease,
            predicted_decrease=predicted_decrease,
            rho=rho,
            rho_bar=rho_bar,
            error=error,
            integral=integral,
            alpha=new_alpha,
            log_alpha=self.log_alpha,
            log_multiplier=log_multiplier,
            accepted=accepted,
            used_fallback_direction=used_fallback,
            grad_dot_direction=grad_dot_direction,
            backtracks=backtracks_taken,
            alpha_next=new_alpha,
            alpha_update_factor=new_alpha / alpha_used,
            trust_region_expanded=trust_region_expanded,
        )

        if accepted:
            return torch.as_tensor(loss_after_value, dtype=loss_before.dtype, device=loss_before.device)
        return loss_before.detach()
