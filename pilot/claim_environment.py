"""Simulated audience for claim-invitation emails. Python 3.9+, standard library only.

Models the three conditions that distinguish this setting from a textbook bandit:

  low base rate      conversion probabilities around one percent are configurable
                     per variant, including the case where all variants are equal

  delayed reward     a conversion arrives some days after the invitation. Only
                     outcomes whose observation window has closed are released to
                     the policy. A pending outcome is NOT a failure: it is simply
                     not yet visible, and counting it as a failure biases every
                     posterior downwards

  batch sending      invitations are dispatched in batches. The policy may update
                     between batches, never within one

Run the diagnostic:
    python claim_environment.py
    python claim_environment.py --rates 0.010 0.013 --batch-size 2000

No real email is sent, no network is accessed, and no personal data is used.
All conversion probabilities are assumptions, not measured population values.
"""

import argparse
import math
import random
from typing import Mapping, Optional, Sequence

# Default scenario: two variants at a one percent base rate with a 30% relative gap.
DEFAULT_RATES = {"A": 0.010, "B": 0.013}
# Days after sending within which a claim still counts as a conversion.
DEFAULT_WINDOW_DAYS = 7
# Days between successive batches.
DEFAULT_BATCH_INTERVAL_DAYS = 7
# Share of converting recipients that act on day 0, 1, 2, ... (geometric-like tail).
DEFAULT_DELAY_WEIGHTS = (0.35, 0.20, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.04)


class ClaimEnvironment:
    """Draw conversion outcomes for chosen variants, with a delay before they are visible.

    The policy must see only `variants` and the outcomes released by
    `collect_matured`. It must never read `_rates`, and it never observes the
    outcome of a variant it did not send.

    One uniform draw is consumed per send regardless of which variant is chosen,
    so two policies run against environments built with the same seed face the
    same recipient-level randomness. This is the method of common random numbers
    and reduces the variance of a paired comparison. It is a simulation device,
    not a claim about real counterfactual outcomes.
    """

    def __init__(
        self,
        rates: Optional[Mapping[str, float]] = None,
        *,
        seed: int = 42,
        window_days: int = DEFAULT_WINDOW_DAYS,
        batch_interval_days: int = DEFAULT_BATCH_INTERVAL_DAYS,
        delay_weights: Sequence[float] = DEFAULT_DELAY_WEIGHTS,
    ):
        source = DEFAULT_RATES if rates is None else rates
        if not source:
            raise ValueError("Provide at least one variant.")
        self._rates = {}
        for variant, value in source.items():
            if not isinstance(variant, str) or not variant:
                raise ValueError("Variant names must be non-empty strings.")
            if isinstance(value, (str, bool)):
                raise ValueError("Conversion probabilities must be numeric.")
            rate = float(value)
            if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise ValueError("Conversion probabilities must be finite and in [0, 1].")
            self._rates[variant] = rate

        # A zero-day window is legitimate: it requires the claim on the send day.
        if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 0:
            raise ValueError("window_days must be a non-negative integer.")
        if (isinstance(batch_interval_days, bool)
                or not isinstance(batch_interval_days, int) or batch_interval_days < 1):
            raise ValueError("batch_interval_days must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Seed must be an integer.")
        weights = [float(w) for w in delay_weights]
        if not weights or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Delay weights must be non-negative and not all zero.")

        self._rng = random.Random(seed)
        self._window_days = window_days
        self._batch_interval_days = batch_interval_days
        self._delay_weights = weights
        # Pending outcomes keyed by the batch index at which they become visible.
        self._pending = {}
        self._sends = 0

        # With a window of W days and batches every I days, an outcome from batch k
        # has closed by the time batch k + ceil(W / I) is allocated.
        self._maturation_lag = math.ceil(window_days / batch_interval_days)

    @property
    def variants(self):
        """Variant names only. Deliberately does not expose the rates."""
        return tuple(self._rates)

    @property
    def maturation_lag(self):
        """Number of batches between sending and the outcome becoming visible."""
        return self._maturation_lag

    @property
    def sends_completed(self):
        return self._sends

    def send(self, variant: str, batch_index: int) -> None:
        """Dispatch one invitation. The outcome is withheld until its window closes."""
        if variant not in self._rates:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {self.variants}.")
        if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
            raise ValueError("batch_index must be a non-negative integer.")

        # Exactly two uniform draws per recipient, consumed whichever variant was
        # chosen and whether or not a conversion occurs. Keeping the number of
        # draws constant is what makes the stream align recipient-by-recipient
        # across policies; drawing the delay only for converters would desynchronise it.
        converts = self._rng.random() < self._rates[variant]
        delay_days = self._draw_delay_days()
        # A conversion arriving after the window closes is recorded as a
        # non-conversion, so the measured rate is the within-window rate.
        reward = int(converts and delay_days <= self._window_days)

        visible_at = batch_index + self._maturation_lag
        self._pending.setdefault(visible_at, []).append((variant, reward))
        self._sends += 1

    def collect_matured(self, batch_index: int):
        """Release every outcome whose observation window has closed by this batch.

        Returns a list of (variant, reward) pairs. Outcomes still pending are not
        returned and must not be treated as failures by the caller.
        """
        if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
            raise ValueError("batch_index must be a non-negative integer.")
        matured = []
        for key in [k for k in self._pending if k <= batch_index]:
            matured.extend(self._pending.pop(key))
        return matured

    def pending_count(self):
        """Number of sends whose outcome has not yet been released."""
        return sum(len(v) for v in self._pending.values())

    def _draw_delay_days(self):
        """Days between the invitation and the claim, for a converting recipient."""
        total = sum(self._delay_weights)
        target = self._rng.random() * total
        cumulative = 0.0
        for day, weight in enumerate(self._delay_weights):
            cumulative += weight
            if target < cumulative:
                return day
        # Residual mass falls beyond the tabulated days and therefore beyond the window.
        return len(self._delay_weights)


def demo():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=[0.010, 0.013],
                        help="Two or three assumed conversion rates.")
    parser.add_argument("--sends", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--batch-interval-days", type=int, default=DEFAULT_BATCH_INTERVAL_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        if len(args.rates) not in (2, 3):
            parser.error("Provide two or three rates.")
        rates = dict(zip(("A", "B", "C"), args.rates))
        environment = ClaimEnvironment(
            rates, seed=args.seed, window_days=args.window_days,
            batch_interval_days=args.batch_interval_days)

        print("SIMULATED CLAIM-INVITATION AUDIENCE")
        print(f"Assumed rates: {rates}")
        print(f"Window: {args.window_days} days; batch interval: "
              f"{args.batch_interval_days} days; maturation lag: "
              f"{environment.maturation_lag} batch(es)\n")

        # Send every variant to an equal share, to show the within-window rates.
        batches = args.sends // args.batch_size
        observed = {v: [0, 0] for v in rates}   # matured trials, matured conversions
        for batch_index in range(batches):
            for position in range(args.batch_size):
                environment.send(list(rates)[position % len(rates)], batch_index)
            # Count only outcomes whose window has closed: a pending send has no
            # label yet and must not appear in either the numerator or denominator.
            for variant, reward in environment.collect_matured(batch_index):
                observed[variant][0] += 1
                observed[variant][1] += reward

        print("Variant   Assumed rate   Matured sends   Conversions   Within-window rate")
        for variant, rate in rates.items():
            trials, conversions = observed[variant]
            share = conversions / trials if trials else float("nan")
            print(f"{variant:7}   {rate:12.4%}   {trials:13,d}   {conversions:11,d}   {share:18.4%}")
        print(f"\nStill pending at the end of the run: {environment.pending_count():,} sends.")
        print("The within-window rate is below the assumed rate because late claims "
              "fall outside the observation window.")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    demo()
