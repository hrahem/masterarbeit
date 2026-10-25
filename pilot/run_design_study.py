"""Design study for the claim-invitation field experiment.

Answers the questions the pre-registration needs:

  1. Feasibility. Can a field run of the available size demonstrate that adaptive
     allocation earns more conversions than a fixed split? The required sample
     size and the expected gain are compared against the run-to-run noise.

  2. Configuration. For two and three variants, across batch sizes, control
     shares and total volumes: conversions gained over a uniform split, the
     probability of identifying the best variant, and the probability of
     committing to a wrong one. The case where all variants are equal is
     included, because a design that cannot be trusted under the null is not
     usable.

  3. Volume. How large the field run must be before the control stream can
     estimate the difference between variants with useful precision.

Run from the project root:

    python pilot/run_design_study.py
    python pilot/run_design_study.py --repetitions 4000 --verify

Uses NumPy for speed. `--verify` cross-checks the vectorised implementation
against the standard-library simulator in claim_environment.py, which is the
reference implementation.

All conversion rates are assumptions. Nothing here is a measured value and
nothing here is evidence about the real campaign.
"""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from claim_environment import (DEFAULT_DELAY_WEIGHTS, DEFAULT_WINDOW_DAYS,
                               DEFAULT_BATCH_INTERVAL_DAYS)

Z_ALPHA = 1.959963984540054      # two-sided 5% significance
Z_POWER = 0.8416212335729143     # 80% power


def within_window_fraction(weights=DEFAULT_DELAY_WEIGHTS, window_days=DEFAULT_WINDOW_DAYS):
    """Share of conversions that arrive before the observation window closes."""
    weights = list(weights)
    return sum(weights[:window_days + 1]) / sum(weights)


def required_sample_size(p1, p2):
    """Per-variant sample size for a two-proportion test at 5% / 80%."""
    return (Z_ALPHA + Z_POWER) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def analytical_power(p1, p2, n_per_arm):
    """Power of the pooled two-proportion test at a given per-variant size."""
    mean_p = (p1 + p2) / 2
    se_null = math.sqrt(2 * mean_p * (1 - mean_p) / n_per_arm)
    se_alt = math.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)
    return 1 - normal_cdf((Z_ALPHA * se_null - (p2 - p1)) / se_alt)


def simulate(rates, *, total_sends, batch_size, control_share, repetitions, rng,
             maturation_lag=1, window_fraction=1.0):
    """Vectorised batched Thompson Sampling beside a uniform control.

    Mirrors run_field_simulation.py: posterior frozen within a batch, outcomes
    withheld for `maturation_lag` batches, control share applied in every batch.
    Returns per-repetition arrays rather than summary statistics, so that
    intervals can be computed by the caller.
    """
    rates = np.asarray(rates, dtype=float)
    effective = rates * window_fraction      # within-window conversion probability
    k = len(rates)
    best = int(np.argmax(rates))
    batches = total_sends // batch_size
    if batches < 1:
        raise ValueError("total_sends must cover at least one batch.")

    n_control = int(round(batch_size * control_share))
    n_adaptive = batch_size - n_control

    alpha = np.ones((repetitions, k))
    beta = np.ones((repetitions, k))
    pending = [np.zeros((repetitions, k, 2)) for _ in range(maturation_lag)]

    adaptive_sends = np.zeros((repetitions, k))
    control_sends = np.zeros((repetitions, k))
    conversions = np.zeros(repetitions)
    control_trials = np.zeros((repetitions, k))
    control_conversions = np.zeros((repetitions, k))

    for _ in range(batches):
        # Release outcomes whose window has closed, then freeze the posterior.
        matured = pending.pop(0)
        alpha += matured[:, :, 0]
        beta += matured[:, :, 1] - matured[:, :, 0]

        if n_adaptive:
            draws = rng.beta(alpha[:, None, :], beta[:, None, :],
                             size=(repetitions, n_adaptive, k))
            choice = np.argmax(draws, axis=2)
            counts = np.stack([(choice == arm).sum(axis=1) for arm in range(k)], axis=1)
        else:
            counts = np.zeros((repetitions, k), dtype=int)

        control = (rng.multinomial(n_control, [1 / k] * k, size=repetitions)
                   if n_control else np.zeros((repetitions, k), dtype=int))

        adaptive_successes = rng.binomial(counts, effective)
        control_successes = rng.binomial(control, effective)

        adaptive_sends += counts
        control_sends += control
        control_trials += control
        control_conversions += control_successes
        conversions += adaptive_successes.sum(axis=1) + control_successes.sum(axis=1)

        trials = counts + control
        successes = adaptive_successes + control_successes
        pending.append(np.stack([successes, trials], axis=2))

    total_per_arm = adaptive_sends + control_sends
    posterior_mean = alpha / (alpha + beta)
    return {
        "conversions": conversions,
        "share_best": total_per_arm[:, best] / total_per_arm.sum(axis=1),
        "picks_best": (np.argmax(posterior_mean, axis=1) == best),
        "majority_wrong": (np.argmax(total_per_arm, axis=1) != best),
        "control_trials": control_trials,
        "control_conversions": control_conversions,
        "uniform_expected": total_sends * effective.mean(),
    }


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    half = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    return values.mean(), values.mean() - half, values.mean() + half


def feasibility_section(rates, total_sends, window_fraction):
    p1, p2 = rates[0] * window_fraction, rates[1] * window_fraction
    n_req = required_sample_size(p1, p2)
    power = analytical_power(p1, p2, total_sends / 2)
    print("1. FEASIBILITY OF A FIELD SUPERIORITY TEST")
    print(f"   Within-window rates: {p1:.4%} vs {p2:.4%} "
          f"(relative {p2 / p1:.2f})")
    print(f"   Required per variant for 5% / 80%: {n_req:,.0f}  "
          f"(total {2 * n_req:,.0f})")
    print(f"   Power at the planned volume of {total_sends:,}: {power:.3f}")
    return {"required_per_variant": n_req, "power_at_planned_volume": power}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rate", type=float, default=0.010)
    parser.add_argument("--lift", type=float, default=1.30,
                        help="Relative advantage of the best variant.")
    parser.add_argument("--total-sends", type=int, default=20000)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--batch-interval-days", type=int, default=DEFAULT_BATCH_INTERVAL_DAYS)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--verify", action="store_true",
                        help="Cross-check against the standard-library simulator.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    fraction = within_window_fraction(window_days=args.window_days)
    lag = math.ceil(args.window_days / args.batch_interval_days)
    base, lift = args.base_rate, args.lift

    print("DESIGN STUDY — ALL RATES ARE ASSUMPTIONS, ALL OUTCOMES SIMULATED")
    print(f"Base rate {base:.3%}; observation window {args.window_days} days "
          f"({fraction:.0%} of conversions arrive inside it); "
          f"maturation lag {lag} batch(es)")
    print(f"{args.repetitions:,} repetitions per configuration\n")

    records = {}
    records["feasibility"] = feasibility_section(
        [base, base * lift], args.total_sends, fraction)

    result = simulate([base, base * lift], total_sends=args.total_sends,
                      batch_size=2000, control_share=0.40,
                      repetitions=args.repetitions, rng=rng,
                      maturation_lag=lag, window_fraction=fraction)
    mean, low, high = mean_ci(result["conversions"])
    uniform = result["uniform_expected"]
    sd = result["conversions"].std(ddof=1)
    print(f"   Uniform split, expected conversions: {uniform:.1f}")
    print(f"   Adaptive, mean conversions:          {mean:.1f}  "
          f"(95% CI of the mean [{low:.1f}, {high:.1f}])")
    print(f"   Expected gain:                       {mean - uniform:+.1f} "
          f"({(mean - uniform) / uniform:+.1%})")
    print(f"   Run-to-run standard deviation:       {sd:.1f}")
    verdict = "SMALLER than" if (mean - uniform) < sd else "larger than"
    print(f"   The gain is {verdict} the noise of a single run.\n")
    records["feasibility"].update({
        "uniform_expected": uniform, "adaptive_mean": mean,
        "gain": mean - uniform, "run_to_run_sd": sd})

    print("2. CONFIGURATION GRID")
    print(f"{'variants':>8} {'scenario':>12} {'batch':>6} {'control':>8} "
          f"{'conversions':>12} {'gain':>8} {'picks best':>11} {'wrong majority':>15}")
    grid = []
    for n_variants in (2, 3):
        scenarios = {
            "equal": [base] * n_variants,
            "lift": [base] * (n_variants - 1) + [base * lift],
        }
        for label, rates in scenarios.items():
            for batch_size in (1000, 2000, 5000):
                for control_share in (0.20, 0.40, 0.50):
                    r = simulate(rates, total_sends=args.total_sends,
                                 batch_size=batch_size, control_share=control_share,
                                 repetitions=args.repetitions, rng=rng,
                                 maturation_lag=lag, window_fraction=fraction)
                    mean_conv = r["conversions"].mean()
                    gain = mean_conv - r["uniform_expected"]
                    row = {
                        "variants": n_variants, "scenario": label,
                        "batch_size": batch_size, "control_share": control_share,
                        "mean_conversions": round(mean_conv, 2),
                        "gain_over_uniform": round(gain, 2),
                        "picks_best_rate": round(float(r["picks_best"].mean()), 4),
                        "wrong_majority_rate": round(float(r["majority_wrong"].mean()), 4),
                        "share_best_p2_5": round(float(np.percentile(r["share_best"], 2.5)), 4),
                        "share_best_p97_5": round(float(np.percentile(r["share_best"], 97.5)), 4),
                    }
                    grid.append(row)
                    print(f"{n_variants:>8} {label:>12} {batch_size:>6} "
                          f"{control_share:>7.0%} {mean_conv:>12.1f} {gain:>+8.1f} "
                          f"{row['picks_best_rate']:>11.1%} "
                          f"{row['wrong_majority_rate']:>15.1%}")
    records["grid"] = grid

    print("\n3. REQUIRED VOLUME (control stream precision, 40% control share)")
    print(f"{'total sends':>12} {'per variant in control':>24} "
          f"{'95% CI halfwidth on the gap':>30}")
    true_gap = base * (lift - 1) * fraction
    volume = []
    for total in (20000, 40000, 60000, 100000, 200000):
        per_arm = total * 0.40 / 2
        p1, p2 = base * fraction, base * lift * fraction
        se = math.sqrt(p1 * (1 - p1) / per_arm + p2 * (1 - p2) / per_arm)
        half = 1.96 * se
        volume.append({"total_sends": total, "per_variant_control": per_arm,
                       "ci_halfwidth_pp": round(100 * half, 4),
                       "true_gap_pp": round(100 * true_gap, 4),
                       "informative": bool(half < true_gap)})
        flag = "" if half < true_gap else "   (wider than the effect)"
        print(f"{total:>12,} {per_arm:>24,.0f} {100 * half:>29.3f}pp{flag}")
    print(f"\n   True gap to be resolved: {100 * true_gap:.3f}pp")
    records["volume"] = volume

    if args.verify:
        print("\n4. VERIFICATION AGAINST THE REFERENCE IMPLEMENTATION")
        from claim_environment import ClaimEnvironment
        from batch_thompson import BatchThompsonSampling
        from run_field_simulation import run_campaign
        reference = []
        for seed in range(60):
            environment = ClaimEnvironment({"A": base, "B": base * lift}, seed=seed,
                                           window_days=args.window_days,
                                           batch_interval_days=args.batch_interval_days)
            policy = BatchThompsonSampling(environment.variants, policy_seed=1000 + seed)
            sends, _ = run_campaign(environment, policy, total_sends=10000,
                                    batch_size=2000, control_share=0.40,
                                    assignment_seed=2000 + seed)
            share = sum(1 for s in sends if s["variant"] == "B") / len(sends)
            reference.append(share)
        fast = simulate([base, base * lift], total_sends=10000, batch_size=2000,
                        control_share=0.40, repetitions=2000, rng=rng,
                        maturation_lag=lag, window_fraction=fraction)
        print(f"   share to best, reference (stdlib, 60 runs): "
              f"{np.mean(reference):.4f}")
        print(f"   share to best, vectorised  (2000 runs):     "
              f"{fast['share_best'].mean():.4f}")
        print("   The two should agree to within the reference's sampling error "
              f"(+/- {1.96 * np.std(reference, ddof=1) / math.sqrt(60):.4f}).")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output_dir.resolve() / f"design_{timestamp}"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "design_study.json").write_text(
        json.dumps({"simulation_only": True, "base_rate": base, "lift": lift,
                    "window_days": args.window_days, "within_window_fraction": fraction,
                    "maturation_lag_batches": lag, "repetitions": args.repetitions,
                    "seed": args.seed, **records}, indent=2, default=float) + "\n",
        encoding="utf-8")
    with (destination / "grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    print(f"\nSaved: {destination}")


if __name__ == "__main__":
    main()
