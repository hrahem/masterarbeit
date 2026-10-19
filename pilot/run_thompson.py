"""Run the basic Thompson Sampling pilot. Python 3.9+, no extra packages.

Place in pilot beside email_environment.py and thompson_sampling.py.
Default: load the same historical click campaign as run_ab_test.py.
--rates 0.01 0.013: use an explicitly hypothetical claim scenario instead.
All responses are immediate. This is not a field-experiment implementation.
"""

import argparse  # Terminal options.
import csv  # Tabular history and summary outputs.
import hashlib  # File fingerprints for reproducibility.
import json  # Structured settings output.
import platform  # Record the Python runtime version.
from datetime import datetime, timezone  # Unique UTC result-directory names.
from pathlib import Path  # Portable Windows/macOS/Linux paths.

from email_environment import CAMPAIGN, EmailEnvironment, load_campaign_rates
from thompson_sampling import run_thompson

# Resolve the user's existing data layout relative to pilot/run_thompson.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = (PROJECT_ROOT / "hubspot_email_campaign" /
                    "hubspot_research_package" / "pilot_ab_baseline.csv")


def write_csv(path, rows):
    """Write dictionaries as rows; use the first row's keys as column names."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    # None defaults allow us to detect conflicting CSV and synthetic arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="Exact CSV campaign name; default: existing paid newsletter.")
    parser.add_argument("--baseline-csv", type=Path, help="Override the existing default data path.")
    parser.add_argument("--rates", nargs="+", type=float,
                        help="Two or three hypothetical claim rates, e.g. 0.01 0.013.")
    parser.add_argument("--recipients", type=int,
                        help="Default: historical deliveries, or 20000 for --rates.")
    parser.add_argument("--seed", type=int, default=42, help="Audience response seed.")
    parser.add_argument("--policy-seed", type=int, default=123, help="Posterior-sampling seed.")
    parser.add_argument("--prior-alpha", type=float, default=1.0, help="Common initial alpha.")
    parser.add_argument("--prior-beta", type=float, default=1.0, help="Common initial beta.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    try:
        # Explicit synthetic mode prevents claim assumptions being mislabeled as data.
        if args.rates is not None:
            if args.campaign is not None or args.baseline_csv is not None:
                raise ValueError("Use --rates OR CSV campaign options, not both.")
            if len(args.rates) not in (2, 3):
                raise ValueError("Provide two or three rates, for variants A/B or A/B/C.")
            probabilities = dict(zip(("A", "B", "C"), args.rates))
            default_budget = 20000
            source = {"mode": "synthetic_claim_scenario", "campaign": None,
                      "outcome": "hypothetical claim", "trial_unit": "simulated email sent",
                      "rates_are_assumptions": True}
        else:
            # Reuse the existing exact-count loader, including its input validation.
            baseline = (args.baseline_csv or DEFAULT_BASELINE).resolve()
            campaign = args.campaign or CAMPAIGN
            probabilities, default_budget = load_campaign_rates(baseline, campaign)
            source = {"mode": "historical_click_calibration", "campaign": campaign,
                      "outcome": "click within seven days", "trial_unit": "simulated delivery",
                      "baseline_csv": str(baseline),
                      "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}
        budget = default_budget if args.recipients is None else args.recipients
        # Only the environment receives rates. The policy receives names and rewards.
        audience = EmailEnvironment(probabilities, seed=args.seed, horizon=budget)
        counts, history, posterior = run_thompson(
            audience, budget, policy_seed=args.policy_seed,
            alpha=args.prior_alpha, beta=args.prior_beta)
        total = sum(item["successes"] for item in counts.values())
        summary = []
        for variant, item in counts.items():
            # A short run may never sample an arm; leave its empirical rate blank.
            rate = item["successes"] / item["trials"] if item["trials"] else None
            summary.append({"variant": variant, **item,
                            "empirical_rate": rate,
                            "allocation_share": item["trials"] / budget,
                            "posterior_alpha": posterior[variant]["alpha"],
                            "posterior_beta": posterior[variant]["beta"],
                            "posterior_mean": posterior[variant]["mean"]})
        # There is no single Beta posterior for the combined adaptive stream.
        summary.append({"variant": "TOTAL", "trials": budget, "successes": total,
                        "empirical_rate": total / budget, "allocation_share": 1.0,
                        "posterior_alpha": None, "posterior_beta": None, "posterior_mean": None})
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = args.output_dir.resolve() / f"ts_{timestamp}"
        destination.mkdir(parents=True, exist_ok=False)  # Preserve previous runs.
        write_csv(destination / "history.csv", history)
        write_csv(destination / "summary.csv", summary)
        # Hash all implementation files, not only the runner.
        hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ("email_environment.py", "thompson_sampling.py", "run_thompson.py")}
        settings = {"simulation_only": True, "strategy": "beta_bernoulli_thompson_sampling",
                    **source, "assumed_probabilities": probabilities,
                    "recipients": budget, "source_default_budget": default_budget,
                    "environment_seed": args.seed, "policy_seed": args.policy_seed,
                    "prior_alpha": args.prior_alpha, "prior_beta": args.prior_beta,
                    "feedback": "immediate", "updates": "after every trial",
                    "assignment_probabilities_logged": False,
                    "python_version": platform.python_version(),
                    "python_implementation": platform.python_implementation(),
                    "code_sha256": hashes, "created_utc": timestamp}
        (destination / "settings.json").write_text(json.dumps(settings, indent=2) + "\n",
                                                   encoding="utf-8")
        print("BASIC THOMPSON SAMPLING — SIMULATED OUTCOMES ONLY")
        print(f"Source: {source['mode']}; outcome: {source['outcome']}")
        print(f"Scenario: {source['campaign'] or 'hypothetical rates'}; rates: {probabilities}")
        print(f"Seeds: audience={args.seed}, policy={args.policy_seed}")
        print("Variant      Trials   Successes   Observed rate   Allocation")
        for row in summary:
            rate_text = "n/a" if row["empirical_rate"] is None else f"{row['empirical_rate']:.2%}"
            print(f"{row['variant']:7} {row['trials']:11,d} {row['successes']:11,d} "
                  f"{rate_text:>15} {row['allocation_share']:12.2%}")
        print(f"Saved: {destination}")
        print("Single run; no superiority claim. No batches, delays, concurrent control, or OPE logging yet.")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


# Importing this file does not start a simulation.
if __name__ == "__main__":
    main()
