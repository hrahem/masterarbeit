"""Batched Thompson Sampling with logged assignment probabilities.

Standard library only. Two things distinguish this from the textbook algorithm:

  batched updates    the posterior is frozen for the duration of a batch, so the
                     allocation within a batch is a fixed probability distribution
                     over variants. Updates happen only at batch boundaries

  logged propensity  the probability with which each variant is selected is
                     computed and recorded for every send. Without this the field
                     log cannot be used for off-policy evaluation afterwards

For two variants the selection probability is computed exactly in closed form.
For three or more it is estimated by Monte Carlo from a dedicated random
generator, so that estimating propensities never perturbs allocation randomness.
"""

import math
import random

# Monte Carlo draws used for propensity estimation when there are three or more variants.
DEFAULT_PROPENSITY_SAMPLES = 20000


def _log_beta(a, b):
    """Natural log of the Beta function, via log-gamma for numerical stability."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def probability_greater(alpha_x, beta_x, alpha_y, beta_y):
    """Exact P(X > Y) for X ~ Beta(alpha_x, beta_x), Y ~ Beta(alpha_y, beta_y).

    Requires integer alpha_x. Uses the standard finite-sum identity, evaluated in
    log space because the individual Beta functions underflow for large counts.
    """
    if alpha_x != int(alpha_x) or alpha_x < 1:
        raise ValueError("Closed form requires a positive integer alpha for X.")
    total = 0.0
    for i in range(int(alpha_x)):
        log_term = (_log_beta(alpha_y + i, beta_y + beta_x)
                    - math.log(beta_x + i)
                    - _log_beta(1 + i, beta_x)
                    - _log_beta(alpha_y, beta_y))
        total += math.exp(log_term)
    # Guard against accumulated floating-point drift just outside [0, 1].
    return min(1.0, max(0.0, total))


class BatchThompsonSampling:
    """Beta-Bernoulli Thompson Sampling that updates only between batches."""

    def __init__(self, variants, *, alpha=1.0, beta=1.0, policy_seed=123,
                 propensity_samples=DEFAULT_PROPENSITY_SAMPLES):
        variants = tuple(variants)
        if len(variants) < 2:
            raise ValueError("Thompson Sampling needs at least two variants.")
        if len(set(variants)) != len(variants):
            raise ValueError("Variant names must be unique.")
        for name, value in (("alpha", alpha), ("beta", beta)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric.")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and strictly positive.")

        self._variants = variants
        self._posterior = {v: [float(alpha), float(beta)] for v in variants}
        # Allocation randomness and propensity estimation use separate generators,
        # so changing the propensity precision cannot change which variants are sent.
        self._rng = random.Random(policy_seed)
        self._propensity_rng = random.Random(policy_seed + 1)
        self._propensity_samples = propensity_samples
        # Selection probabilities held fixed for the current batch.
        self._frozen = self.selection_probabilities()

    @property
    def variants(self):
        return self._variants

    def posterior(self):
        """Current Beta parameters and means, as a copy."""
        return {v: {"alpha": a, "beta": b, "mean": a / (a + b)}
                for v, (a, b) in self._posterior.items()}

    def selection_probabilities(self):
        """P(variant is the best) under the current posterior, one entry per variant."""
        if len(self._variants) == 2:
            first, second = self._variants
            a1, b1 = self._posterior[first]
            a2, b2 = self._posterior[second]
            p_first = probability_greater(a1, b1, a2, b2)
            return {first: p_first, second: 1.0 - p_first}

        counts = {v: 0 for v in self._variants}
        for _ in range(self._propensity_samples):
            draws = {v: self._propensity_rng.betavariate(*self._posterior[v])
                     for v in self._variants}
            counts[max(draws, key=draws.get)] += 1
        return {v: counts[v] / self._propensity_samples for v in self._variants}

    def begin_batch(self):
        """Freeze the posterior for a new batch and return its selection probabilities."""
        self._frozen = self.selection_probabilities()
        return dict(self._frozen)

    def select(self):
        """Choose one variant by sampling from the frozen posterior."""
        draws = {v: self._rng.betavariate(*self._posterior[v]) for v in self._variants}
        return max(draws, key=draws.get)

    def frozen_probabilities(self):
        """Selection probabilities in force for the current batch."""
        return dict(self._frozen)

    def update(self, variant, reward):
        """Record one matured outcome. Call only between batches."""
        if variant not in self._posterior:
            raise ValueError(f"Unknown variant {variant!r}.")
        if isinstance(reward, bool) or reward not in (0, 1):
            raise ValueError("Reward must be the integer 0 or 1.")
        self._posterior[variant][0] += reward
        self._posterior[variant][1] += 1 - reward


def uniform_probabilities(variants):
    """Allocation probabilities of the concurrent random control stream."""
    variants = tuple(variants)
    return {v: 1.0 / len(variants) for v in variants}


def logging_probability(variant, control_share, control_probabilities,
                        policy_probabilities):
    """Overall probability that a send received `variant`, across both streams.

    This is the propensity that must be written to the log. It mixes the control
    stream and the adaptive stream, because a recipient is assigned to one or the
    other at random and an off-policy estimator needs the marginal probability.
    """
    return (control_share * control_probabilities[variant]
            + (1.0 - control_share) * policy_probabilities[variant])
