"""Basic Beta-Bernoulli Thompson Sampling, using Python's standard library.

This policy knows only variant names and observed binary rewards. It never
receives the environment's true probabilities. See THOMPSON_SAMPLING.md.
"""

import math  # Reject nonfinite prior parameters.
import random  # Use a dedicated generator, separate from audience randomness.


class ThompsonSampling:
    """Maintain one Beta distribution per variant and update chosen arms only."""

    def __init__(self, variants, *, seed=123, alpha=1.0, beta=1.0):
        # Materialize names once so their ordering is stable for reproducibility.
        self.variants = tuple(variants)
        # Reject an empty action set, invalid labels, or duplicate labels.
        if not self.variants or any(not isinstance(v, str) or not v for v in self.variants):
            raise ValueError("Provide nonempty string variant names.")
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("Variant names must be unique.")
        # Beta distributions require two strictly positive, finite parameters.
        for name, value in (("alpha", alpha), ("beta", beta)):
            if isinstance(value, (bool, str)) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive finite number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Seed must be an integer.")
        # The same prior is used for each variant; no historical outcomes leak in.
        self._alpha = {v: float(alpha) for v in self.variants}
        self._beta = {v: float(beta) for v in self.variants}
        self._rng = random.Random(seed)

    def select(self):
        """Return (chosen_variant, sampled_rates), before seeing this reward.

        Sampled rates are diagnostic draws, NOT action assignment probabilities.
        Sampling does not update the posterior; only update() does that.
        """
        # Draw a plausible success rate from each arm's current posterior.
        draws = {v: self._rng.betavariate(self._alpha[v], self._beta[v])
                 for v in self.variants}
        # Choose the largest draw, rather than always using the posterior mean.
        highest = max(draws.values())
        candidates = [v for v in self.variants if draws[v] == highest]
        # Exact floating-point ties are rare; break them randomly without A bias.
        chosen = candidates[0] if len(candidates) == 1 else self._rng.choice(candidates)
        return chosen, draws

    def update(self, variant, reward):
        """Observe one finalized 0/1 reward for the chosen variant.

        Pending outcomes must not be passed here. The caller is responsible for
        updating once per observation; this basic policy does not track send IDs.
        """
        if variant not in self._alpha:
            raise ValueError(f"Unknown variant: {variant!r}")
        # In particular, None (pending) is invalid, not a no-conversion outcome.
        if isinstance(reward, bool) or not isinstance(reward, int) or reward not in (0, 1):
            raise ValueError("Reward must be integer 0 or 1; pending is not failure.")
        # Success increments alpha; failure increments beta; other arms stay unchanged.
        self._alpha[variant] += reward
        self._beta[variant] += 1 - reward

    def posterior(self):
        """Return a copy of the parameters and posterior mean for each variant."""
        return {v: {"alpha": self._alpha[v], "beta": self._beta[v],
                    "mean": self._alpha[v] / (self._alpha[v] + self._beta[v])}
                for v in self.variants}


def run_thompson(audience, recipients, *, policy_seed=123, alpha=1.0, beta=1.0):
    """Run one immediate-feedback simulation; return counts, history, posteriors.

    The audience interface requires only .variants and .send(chosen_variant).
    The simulation orchestrator constructs the environment; the policy cannot
    inspect its probabilities. A fresh policy is created for every call.
    """
    if isinstance(recipients, bool) or not isinstance(recipients, int) or recipients < 1:
        raise ValueError("Recipients must be a positive integer.")
    policy = ThompsonSampling(audience.variants, seed=policy_seed, alpha=alpha, beta=beta)
    counts = {v: {"trials": 0, "successes": 0} for v in policy.variants}
    history = []  # One audit record per simulated trial.
    total = 0
    for step in range(1, recipients + 1):
        chosen, draws = policy.select()  # Choose before obtaining the reward.
        reward = audience.send(chosen)  # Reveal only the chosen arm's outcome.
        policy.update(chosen, reward)  # Immediate feedback in this first version.
        counts[chosen]["trials"] += 1
        counts[chosen]["successes"] += reward
        total += reward
        row = {"step": step, "variant": chosen, "reward": reward,
               "cumulative_successes": total}
        # Keep pre-outcome samples and post-update beliefs distinct in the log.
        for v, state in policy.posterior().items():
            row[f"sampled_rate_{v}"] = draws[v]
            row[f"alpha_after_{v}"] = state["alpha"]
            row[f"beta_after_{v}"] = state["beta"]
        history.append(row)
    return counts, history, policy.posterior()
