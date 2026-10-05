"""Task 1: a simulated email audience (Python 3.9+, no extra packages).

Run the demonstration:
    python email_environment.py
    python email_environment.py --seed 7 --recipients 10000

Optionally read the previously prepared campaign summary:
    python email_environment.py --baseline-csv pilot_ab_baseline.csv

Use from the next components:
    from email_environment import EmailEnvironment
    audience = EmailEnvironment(seed=42, horizon=5627)
    reward = audience.send("A")  # 1 = simulated click, 0 = no click

Historical calibration: DE_NL-CW12-25_DACH-DE-PAIDvA, cleaned pilot,
45 clicks / 2815 deliveries for A; 73 / 2812 for B (seven-day outcomes).
These estimates define a hypothetical stationary environment. They are NOT
known real-world probabilities, a trained predictor, or held-out evaluation.

This initial component reveals completed outcomes immediately. Treat it as an
idealized benchmark. Before realistic email experiments, add a feedback queue
so clicks arrive after delays and no-click labels mature after seven days.
No real email is sent, no network is accessed, and no personal data is needed.
"""

# Read command-line options such as --campaign and --seed.
import argparse
# Read and write comma-separated tables using column names.
import csv
# Use isfinite to reject NaN and infinity as invalid probabilities.
import math
# Generate repeatable pseudo-random numbers; no external package is needed.
import random
# Represent file paths portably on Windows, macOS, and Linux.
from pathlib import Path
# Type hints document dictionary-like inputs and values that may be None.
from typing import Mapping, Optional


# Default campaign name; another campaign can be selected with --campaign and a CSV.
CAMPAIGN = "DE_NL-CW12-25_DACH-DE-PAIDvA"
# Fallback empirical probabilities: seven-day unique clicks divided by deliveries.
DEFAULT_PROBABILITIES = {"A": 45 / 2815, "B": 73 / 2812}
# Fallback recipient budget: the sum of cleaned A and B deliveries.
DEFAULT_HORIZON = 2815 + 2812


# A class bundles the simulated audience state and its response-generating methods.
class EmailEnvironment:
    """Return a Bernoulli click outcome for each chosen email variant.

    Each call represents a new simulated recipient, not a repeat send to a
    real person. Outcomes are independent over time with constant arm rates.
    There are no recipient features, fatigue, conversions, or delivery failures.
    The simulated population is conditional on delivery, as in calibration.

    Algorithms should see only `variants`, then call `send(chosen_variant)`.
    They must not inspect `_probabilities` or receive unchosen outcomes.

    Use a fresh environment with the SAME seed for each policy in a paired
    comparison, and a new seed for each repetition. One uniform draw is used
    per recipient regardless of action. This couples policies by recipient
    for variance reduction; it is a simulation choice, not an observed joint
    distribution of real counterfactual outcomes. Policy randomness must use
    its OWN random generator, never this environment's generator.
    """

    # Initialize a fresh audience; self refers to this particular environment instance.
    def __init__(
        self,
        # Accept variant-to-probability values; None means use the built-in example.
        probabilities: Optional[Mapping[str, float]] = None,
        # Arguments after * must be named, for example seed=42 rather than a positional value.
        *,
        # Start the environment random generator from this integer; 42 has no scientific meaning.
        seed: int = 42,
        # Maximum number of simulated deliveries; None means no enforced limit.
        horizon: Optional[int] = None,
    ):
        # Choose supplied rates or fallback rates. An empty dictionary is not a valid fallback.
        source = DEFAULT_PROBABILITIES if probabilities is None else probabilities
        # Reject an empty set of variants.
        if not source:
            # Stop early because a simulation needs at least one possible action.
            raise ValueError("Provide at least one email variant.")
        # Store a validated copy; the caller cannot alter it by editing the original dictionary.
        self._probabilities = {}
        # Validate each variant name and its probability.
        for variant, value in source.items():
            # Require a nonempty string label, such as A or B.
            if not isinstance(variant, str) or not variant:
                # Explain the invalid variant label to the caller.
                raise ValueError("Variant names must be non-empty strings.")
            # Reject text and booleans even though Python can convert some of them to numbers.
            if isinstance(value, (str, bool)):
                # Require a genuinely numeric input.
                raise ValueError("Click probabilities must be numeric.")
            # Convert the numeric input to a floating-point probability.
            p = float(value)
            # Valid probabilities must be finite and lie between zero and one, inclusive.
            if not math.isfinite(p) or not 0 <= p <= 1:
                # Report an impossible or undefined probability.
                raise ValueError("Click probabilities must be finite and between 0 and 1.")
            # Remember this variant's constant probability. Policies must not inspect these values.
            self._probabilities[variant] = p
        # Require an integer seed; explicitly exclude bool, which subclasses int in Python.
        if isinstance(seed, bool) or not isinstance(seed, int):
            # Report an unsupported seed value.
            raise ValueError("Seed must be an integer.")
        # Validate a delivery budget when one was supplied.
        if horizon is not None and (
            # A budget must be a positive integer and cannot be a boolean.
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1
        ):
            # Reject invalid delivery budgets before generating any outcomes.
            raise ValueError("Horizon must be a positive integer or None.")
        # Create a private response generator, separate from the allocation generator.
        self._rng = random.Random(seed)
        # Store the delivery limit.
        self._horizon = horizon
        # Start the completed-delivery counter at zero.
        self._round = 0

    # Expose the following method as a readable attribute without parentheses.
    @property
    # Let the policy discover action names without exposing their probabilities.
    def variants(self):
        # Return immutable variant names, for example (A, B), in dictionary insertion order.
        return tuple(self._probabilities)

    # Expose the following method as a readable attribute without parentheses.
    @property
    # Provide the number of simulated deliveries already completed.
    def rounds_completed(self):
        # Read the counter without changing it.
        return self._round

    # Take one chosen variant and return one binary click outcome (integer 0 or 1).
    def send(self, variant: str) -> int:
        """Choose first, then reveal ONLY the chosen variant's binary reward."""
        # Check the action before consuming randomness or a delivery slot.
        if variant not in self._probabilities:
            # Show valid variants when an invalid one is chosen; f inserts expressions into text.
            raise ValueError(f"Unknown variant {variant!r}. Choose from {self.variants}.")
        # Enforce the recipient limit if one is configured.
        if self._horizon is not None and self._round >= self._horizon:
            # Prevent extra deliveries once the campaign budget has been used.
            raise RuntimeError("The simulated campaign has reached its recipient budget.")

        # A uniform random number falls below p with probability p.
        # Draw one uniform number u between 0 (inclusive) and 1 (exclusive).
        random_number = self._rng.random()
        # Return 1 when u < p, otherwise 0: a Bernoulli outcome with click probability p.
        click = int(random_number < self._probabilities[variant])
        # Count this delivery regardless of whether it produced a click.
        self._round += 1
        # Reveal only the outcome for the chosen variant, never the unchosen variant.
        return click


# Read one campaign from the prepared summary; return its rates and delivery budget.
def load_campaign_rates(path: Path, campaign: str = CAMPAIGN):
    """Load exact ratios from counts, not rounded rates, in the audit summary."""
    # Open safely; utf-8-sig handles a possible Excel BOM, and with closes the file.
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        # DictReader makes each row a dictionary; keep only exact campaign-name matches.
        matches = [r for r in csv.DictReader(f) if r.get("campaign_name") == campaign]
    # Require exactly one row so missing or duplicate campaigns cannot be silently used.
    if len(matches) != 1:
        # Report the selected name and how many matching rows were found.
        raise ValueError(f"Expected exactly one row for {campaign!r}; found {len(matches)}.")
    # Select the single matching dictionary; Python indices start at zero.
    row = matches[0]
    # Prepare the A/B probability mapping to return.
    probabilities = {}
    # Prepare a running total of deliveries across A and B.
    horizon = 0
    # Read the same fields for each of the two variants.
    for variant in ("A", "B"):
        # Build the column name (A_delivered or B_delivered) and parse its integer count.
        delivered = int(row[f"{variant}_delivered"])
        # Read the count of delivered messages with a click within seven days.
        clicks = int(row[f"{variant}_clicks_7d"])
        # Require a positive denominator and a valid binary-outcome click count.
        if delivered <= 0 or not 0 <= clicks <= delivered:
            # Reject inconsistent source counts rather than constructing misleading rates.
            raise ValueError(f"Invalid delivery/click counts for variant {variant}.")
        # Use exact source counts; do not use the CSV's rounded rate column.
        probabilities[variant] = clicks / delivered
        # Add this variant's historical deliveries to the default simulation budget.
        horizon += delivered
    # Return two values; the runner receives them by tuple unpacking.
    return probabilities, horizon


# Provide an optional standalone diagnostic; the experiment runner imports the class.
def demo():
    # __doc__ is the opening module description, reused in --help.
    parser = argparse.ArgumentParser(description=__doc__)
    # The environment seed controls random click responses, not the click probabilities.
    parser.add_argument("--seed", type=int, default=42)
    # Optionally override the simulated audience size; this does not add real evidence.
    parser.add_argument("--recipients", type=int, default=None,
                        help="Recipient budget per diagnostic run (default: campaign size).")
    # Accept a CSV path; type=Path converts the supplied text to a Path object.
    parser.add_argument("--baseline-csv", type=Path,
                        help="Optional pilot_ab_baseline.csv from the cleaned package.")
    # Select an exact campaign_name; use CAMPAIGN when this option is omitted.
    parser.add_argument("--campaign", default=CAMPAIGN)
    # Parse terminal arguments; access each option as an attribute such as args.seed.
    args = parser.parse_args()
    # A different campaign cannot use the hard-coded fallback counts.
    if args.campaign != CAMPAIGN and args.baseline_csv is None:
        # Ask for the summary CSV when selecting a nondefault campaign.
        parser.error("Selecting another campaign requires --baseline-csv.")
    # Run the main operation; report expected input or filesystem errors below.
    try:
        # Load the selected campaign when a CSV argument was provided.
        if args.baseline_csv:
            # Unpack the campaign rates and its historical delivery budget.
            probabilities, default_n = load_campaign_rates(args.baseline_csv, args.campaign)
        # Otherwise use the built-in example, which belongs only to the default campaign.
        else:
            # Copy the fallback mapping and use its matching budget.
            probabilities, default_n = dict(DEFAULT_PROBABILITIES), DEFAULT_HORIZON
        # Prefer an explicit recipient override; otherwise use the historical budget.
        n = default_n if args.recipients is None else args.recipients
        # Reject a zero or negative demonstration size.
        if n <= 0:
            # Report that the demonstration needs a positive audience size.
            parser.error("--recipients must be positive.")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print("TASK 1: SIMULATED AUDIENCE")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print(f"Calibration campaign: {args.campaign}")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print(f"Recipients per diagnostic run: {n:,}; seed: {args.seed}")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print("Immediate feedback; stationary probabilities; simulated outcomes only.\n")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print("Variant   Assumed probability   Simulated clicks   Simulated rate")
        # Run one all-A and one all-B scenario separately; this is not split allocation.
        for variant, p in probabilities.items():
            # Reset the audience for each scenario using the same sequence of uniform draws.
            audience = EmailEnvironment(probabilities, seed=args.seed, horizon=n)
            # Generate n responses for this variant and sum their 0/1 outcomes; _ is unused.
            clicks = sum(audience.send(variant) for _ in range(n))
            # Display diagnostic or result text; f-strings insert values and format percentages.
            print(f"{variant:7}   {p:18.4%}   {clicks:16,d}   {clicks/n:14.4%}")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print("\nThese are two separate all-A/all-B diagnostic runs, not an A/B test.")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print("Next component: a fixed randomized A/B allocator using audience.send().")
    # Catch file errors, invalid values, and missing CSV columns.
    except (OSError, ValueError, KeyError) as exc:
        # Print a readable error and usage instructions, then exit unsuccessfully.
        parser.error(str(exc))


# Execute the entry point only when run directly; importing the file does not run it.
if __name__ == "__main__":
    # Start the standalone demonstration.
    demo()
