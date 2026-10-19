"""Replication and sensitivity study for the pilot comparison.

Place beside email_environment.py, thompson_sampling.py and run_ab_test.py,
then run from the project root:

    python pilot/run_replication.py
    python pilot/run_replication.py --repetitions 2000 --sensitivity-repetitions 500

This repeats the single-run comparison many times with a fresh environment seed
per repetition, paired across policies via common random numbers, and then
repeats it again for a range of smaller assumed effect sizes. It produces the
numbers reported in Tables 5.4 and 5.5 of the thesis.

No real email is sent and no network is accessed. All outcomes are simulated.
"""

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from email_environment import CAMPAIGN, EmailEnvironment, load_campaign_rates
from thompson_sampling import run_thompson
from run_ab_test import run_ab

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = (
    PROJECT_ROOT / "hubspot_email_campaign" / "hubspot_research_package"
    / "pilot_ab_baseline.csv"
)
# Relative effects examined in the sensitivity study; 1.0 means no true difference.
SENSITIVITY_LIFTS = (1.0, 1.1, 1.2, 1.3, 1.4)


def normal_cdf(x):
    """Standard normal CDF via the error function; avoids a SciPy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_z(successes_a, trials_a, successes_b, trials_b):
    """Pooled-variance two-proportion z-test. Returns (z, two-sided p)."""
    if trials_a == 0 or trials_b == 0:
        return 0.0, 1.0
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / trials_a + 1 / trials_b))
    if se == 0:
        return 0.0, 1.0
    z = (successes_b / trials_b - successes_a / trials_a) / se
    return z, 2 * (1 - normal_cdf(abs(z)))


def required_sample_size(p_a, p_b, alpha=0.05, power=0.80):
    """Per-arm sample size for a two-proportion test at the given power."""
    z_alpha = 1.959963984540054 if alpha == 0.05 else None
    z_power = 0.8416212335729143 if power == 0.80 else None
    if z_alpha is None or z_power is None:
        raise ValueError("Only alpha=0.05 and power=0.80 are tabulated here.")
    variance = p_a * (1 - p_a) + p_b * (1 - p_b)
    return (z_alpha + z_power) ** 2 * variance / (p_b - p_a) ** 2


def analytical_power(p_a, p_b, n_per_arm, alpha=0.05):
    """Power of the pooled two-proportion test at a given per-arm sample size."""
    z_alpha = 1.959963984540054
    mean_p = (p_a + p_b) / 2
    se_null = math.sqrt(2 * mean_p * (1 - mean_p) / n_per_arm)
    se_alt = math.sqrt(p_a * (1 - p_a) / n_per_arm + p_b * (1 - p_b) / n_per_arm)
    return 1 - normal_cdf((z_alpha * se_null - (p_b - p_a)) / se_alt)


def one_repetition(probabilities, horizon, environment_seed, policy_seed):
    """Run both policies against identically seeded environments (paired design)."""
    ts_counts, _, _ = run_thompson(
        EmailEnvironment(probabilities, seed=environment_seed, horizon=horizon),
        horizon, policy_seed=policy_seed)
    ab_counts, _ = run_ab(
        EmailEnvironment(probabilities, seed=environment_seed, horizon=horizon),
        horizon, policy_seed)
    return ts_counts, ab_counts


def replicate(probabilities, horizon, repetitions, seed_offset=1000):
    """Repeat the paired comparison and collect per-repetition outcomes."""
    best_arm = max(probabilities, key=probabilities.get)
    oracle = horizon * probabilities[best_arm]
    record = {
        "ts_clicks": [], "ab_clicks": [], "ts_share_best": [],
        "ts_regret": [], "ab_regret": [],
        "ts_wrong_arm": 0, "ab_wrong_winner": 0, "ab_significant": 0,
    }
    for index in range(repetitions):
        ts_counts, ab_counts = one_repetition(
            probabilities, horizon,
            environment_seed=seed_offset + index,
            policy_seed=seed_offset + repetitions + index)

        ts_total = sum(item["successes"] for item in ts_counts.values())
        ab_total = sum(item["clicks"] for item in ab_counts.values())
        record["ts_clicks"].append(ts_total)
        record["ab_clicks"].append(ab_total)

        share = ts_counts[best_arm]["trials"] / horizon
        record["ts_share_best"].append(share)
        if share < 0.5:
            record["ts_wrong_arm"] += 1

        # Expected regret uses the known arm rates rather than realized rewards.
        ts_expected = sum(ts_counts[a]["trials"] * probabilities[a] for a in probabilities)
        ab_expected = sum(ab_counts[a]["deliveries"] * probabilities[a] for a in probabilities)
        record["ts_regret"].append(oracle - ts_expected)
        record["ab_regret"].append(oracle - ab_expected)

        rate_a = ab_counts["A"]["clicks"] / ab_counts["A"]["deliveries"]
        rate_b = ab_counts["B"]["clicks"] / ab_counts["B"]["deliveries"]
        observed_winner = "B" if rate_b > rate_a else "A"
        if observed_winner != best_arm:
            record["ab_wrong_winner"] += 1
        _, p_value = two_proportion_z(
            ab_counts["A"]["clicks"], ab_counts["A"]["deliveries"],
            ab_counts["B"]["clicks"], ab_counts["B"]["deliveries"])
        if p_value < 0.05 and observed_winner == best_arm:
            record["ab_significant"] += 1
    return record


def summarize(record, repetitions):
    """Reduce per-repetition outcomes to the statistics reported in the thesis."""
    differences = [t - a for t, a in zip(record["ts_clicks"], record["ab_clicks"])]
    mean_difference = statistics.mean(differences)
    half_width = 1.96 * statistics.stdev(differences) / math.sqrt(repetitions)
    return {
        "repetitions": repetitions,
        "ts_mean_clicks": statistics.mean(record["ts_clicks"]),
        "ts_sd_clicks": statistics.stdev(record["ts_clicks"]),
        "ab_mean_clicks": statistics.mean(record["ab_clicks"]),
        "ab_sd_clicks": statistics.stdev(record["ab_clicks"]),
        "mean_difference": mean_difference,
        "difference_ci_low": mean_difference - half_width,
        "difference_ci_high": mean_difference + half_width,
        "relative_gain": mean_difference / statistics.mean(record["ab_clicks"]),
        "p_ts_at_least_ab": sum(1 for d in differences if d >= 0) / repetitions,
        "ts_mean_regret": statistics.mean(record["ts_regret"]),
        "ab_mean_regret": statistics.mean(record["ab_regret"]),
        "ts_mean_share_best": statistics.mean(record["ts_share_best"]),
        "ts_min_share_best": min(record["ts_share_best"]),
        "ts_max_share_best": max(record["ts_share_best"]),
        "ts_wrong_arm_rate": record["ts_wrong_arm"] / repetitions,
        "ab_wrong_winner_rate": record["ab_wrong_winner"] / repetitions,
        "ab_significant_rate": record["ab_significant"] / repetitions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--campaign", default=CAMPAIGN)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--sensitivity-repetitions", type=int, default=500)
    parser.add_argument("--recipients", type=int, help="Default: campaign delivery count.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    try:
        baseline = args.baseline_csv.resolve()
        if not baseline.is_file():
            raise ValueError(f"Campaign CSV not found: {baseline}")
        probabilities, campaign_budget = load_campaign_rates(baseline, args.campaign)
        horizon = campaign_budget if args.recipients is None else args.recipients
        p_a, p_b = probabilities["A"], probabilities["B"]

        print("REPLICATION STUDY — SIMULATED OUTCOMES ONLY")
        print(f"Campaign: {args.campaign}")
        print(f"Assumed rates: A={p_a:.5f}, B={p_b:.5f} (relative {p_b / p_a:.3f})")
        print(f"Horizon: {horizon:,} deliveries; repetitions: {args.repetitions:,}\n")

        n_required = required_sample_size(p_a, p_b)
        power = analytical_power(p_a, p_b, horizon / 2)
        print(f"Per-arm n for 80% power: {n_required:,.0f} (total {2 * n_required:,.0f})")
        print(f"Analytical power at n={horizon / 2:,.0f} per arm: {power:.3f}\n")

        main_record = replicate(probabilities, horizon, args.repetitions)
        result = summarize(main_record, args.repetitions)

        print("                        A/B baseline   Thompson Sampling")
        print(f"Mean clicks             {result['ab_mean_clicks']:12.2f}   {result['ts_mean_clicks']:17.2f}")
        print(f"Standard deviation      {result['ab_sd_clicks']:12.2f}   {result['ts_sd_clicks']:17.2f}")
        print(f"Mean expected regret    {result['ab_mean_regret']:12.2f}   {result['ts_mean_regret']:17.2f}")
        print(f"Mean share to best arm  {0.5:12.4f}   {result['ts_mean_share_best']:17.4f}")
        print(f"Converged on inferior   {result['ab_wrong_winner_rate']:12.4f}   {result['ts_wrong_arm_rate']:17.4f}")
        print(f"\nPaired difference: {result['mean_difference']:+.2f} clicks "
              f"(95% CI [{result['difference_ci_low']:.2f}, {result['difference_ci_high']:.2f}]), "
              f"relative {result['relative_gain']:+.1%}")
        print(f"Thompson Sampling at least matches A/B in {result['p_ts_at_least_ab']:.1%} of repetitions")
        print(f"A/B reached p<0.05 in the correct direction in {result['ab_significant_rate']:.2%} "
              f"(analytical power {power:.3f})\n")

        print("SENSITIVITY TO THE ASSUMED EFFECT SIZE")
        print("lift   p_B       A/B clicks   TS clicks   gain      inferior arm")
        sensitivity_rows = []
        for lift in SENSITIVITY_LIFTS + (p_b / p_a,):
            scenario = {"A": p_a, "B": p_a * lift}
            record = replicate(scenario, horizon, args.sensitivity_repetitions,
                               seed_offset=50000)
            ts_mean = statistics.mean(record["ts_clicks"])
            ab_mean = statistics.mean(record["ab_clicks"])
            gain = (ts_mean - ab_mean) / ab_mean if ab_mean else 0.0
            wrong = record["ts_wrong_arm"] / args.sensitivity_repetitions
            sensitivity_rows.append({
                "lift": lift, "p_b": p_a * lift, "ab_mean_clicks": ab_mean,
                "ts_mean_clicks": ts_mean, "relative_gain": gain,
                "ts_wrong_arm_rate": wrong,
            })
            print(f"{lift:5.3f}  {p_a * lift:.5f}  {ab_mean:10.2f}  {ts_mean:10.2f}  "
                  f"{gain:+7.1%}  {wrong:12.1%}")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = args.output_dir.resolve() / f"replication_{timestamp}"
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "replication.json").write_text(
            json.dumps({
                "simulation_only": True,
                "campaign": args.campaign,
                "assumed_probabilities": probabilities,
                "horizon": horizon,
                "required_n_per_arm_80pct": n_required,
                "analytical_power": power,
                "main": result,
            }, indent=2) + "\n", encoding="utf-8")
        with (destination / "sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]))
            writer.writeheader()
            writer.writerows(sensitivity_rows)

        print(f"\nSaved: {destination}")
        print("Conditional on the assumed stationary rates; not a real-world improvement claim.")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
