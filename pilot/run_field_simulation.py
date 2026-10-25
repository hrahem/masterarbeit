"""Simulate one field campaign: batched Thompson Sampling beside a uniform control.

Standard library only. Run from the project root:

    python pilot/run_field_simulation.py
    python pilot/run_field_simulation.py --rates 0.010 0.010 --control-share 0.4

Each batch splits its recipients between a uniform random control stream and the
adaptive stream. Every send is written to the log with the variant it received,
the stream it came from, and the probability with which that variant was selected.
That propensity column is what makes the log usable for off-policy evaluation; a
log without it cannot be reanalysed for any policy other than the one that ran.

This produces the per-send log format the real field experiment must also produce.
No real email is sent and all outcomes are simulated.
"""

import argparse
import csv
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

from claim_environment import ClaimEnvironment, DEFAULT_WINDOW_DAYS, DEFAULT_BATCH_INTERVAL_DAYS
from batch_thompson import (BatchThompsonSampling, uniform_probabilities,
                            logging_probability)

CONTROL = "control"
ADAPTIVE = "adaptive"


def run_campaign(environment, policy, *, total_sends, batch_size, control_share,
                 assignment_seed=321):
    """Run one campaign and return per-send records plus per-batch summaries.

    The control share is applied within every batch, so the unbiased stream
    accumulates throughout rather than being concentrated at the start.
    """
    if isinstance(total_sends, bool) or not isinstance(total_sends, int) or total_sends < 1:
        raise ValueError("total_sends must be a positive integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not 0.0 <= control_share <= 1.0:
        raise ValueError("control_share must lie in [0, 1].")

    assignment_rng = random.Random(assignment_seed)
    control_probabilities = uniform_probabilities(environment.variants)
    batches = total_sends // batch_size
    if batches < 1:
        raise ValueError("total_sends must be at least one batch.")

    sends, batch_rows = [], []
    for batch_index in range(batches):
        # Outcomes whose window has closed by the time this batch is allocated.
        # A send in batch k becomes visible at batch k + maturation_lag, so asking
        # for batch_index here gives exactly the intended lag and no more.
        matured = environment.collect_matured(batch_index)
        for variant, reward in matured:
            policy.update(variant, reward)

        # Freeze the posterior: the allocation is constant for the whole batch.
        policy_probabilities = policy.begin_batch()

        control_count = int(round(batch_size * control_share))
        streams = [CONTROL] * control_count + [ADAPTIVE] * (batch_size - control_count)
        assignment_rng.shuffle(streams)

        for position, stream in enumerate(streams):
            if stream == CONTROL:
                variant = assignment_rng.choice(environment.variants)
            else:
                variant = policy.select()
            propensity = logging_probability(
                variant, control_share, control_probabilities, policy_probabilities)
            environment.send(variant, batch_index)
            sends.append({
                "batch": batch_index,
                "position": position,
                "stream": stream,
                "variant": variant,
                "assignment_probability": round(propensity, 8),
            })

        batch_rows.append({
            "batch": batch_index,
            "sends": batch_size,
            "control_sends": control_count,
            "matured_outcomes_used": len(matured),
            "pending_at_batch_end": environment.pending_count(),
            **{f"p_select_{v}": round(policy_probabilities[v], 6)
               for v in environment.variants},
            **{f"posterior_mean_{v}": round(policy.posterior()[v]["mean"], 8)
               for v in environment.variants},
        })

    # Release everything still pending so the final accounting is complete.
    for variant, reward in environment.collect_matured(batches + environment.maturation_lag):
        policy.update(variant, reward)
    return sends, batch_rows


def summarize(sends, policy, environment):
    """Per-variant and per-stream totals. Conversions come from the policy posterior."""
    posterior = policy.posterior()
    rows = []
    for variant in environment.variants:
        sent = sum(1 for s in sends if s["variant"] == variant)
        control_sent = sum(1 for s in sends
                           if s["variant"] == variant and s["stream"] == CONTROL)
        # alpha and beta started at the prior, so subtract it to recover counts.
        conversions = posterior[variant]["alpha"] - 1
        trials = conversions + posterior[variant]["beta"] - 1
        rows.append({
            "variant": variant,
            "sent": sent,
            "control_sent": control_sent,
            "adaptive_sent": sent - control_sent,
            "allocation_share": sent / len(sends),
            "matured_trials": int(trials),
            "matured_conversions": int(conversions),
            "observed_rate": conversions / trials if trials else None,
            "posterior_mean": posterior[variant]["mean"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=[0.010, 0.013],
                        help="Two or three assumed conversion rates.")
    parser.add_argument("--total-sends", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--control-share", type=float, default=0.40)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--batch-interval-days", type=int, default=DEFAULT_BATCH_INTERVAL_DAYS)
    parser.add_argument("--seed", type=int, default=42, help="Audience randomness.")
    parser.add_argument("--policy-seed", type=int, default=123, help="Posterior sampling.")
    parser.add_argument("--assignment-seed", type=int, default=321, help="Stream assignment.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    try:
        if len(args.rates) not in (2, 3):
            parser.error("Provide two or three rates.")
        rates = dict(zip(("A", "B", "C"), args.rates))

        environment = ClaimEnvironment(
            rates, seed=args.seed, window_days=args.window_days,
            batch_interval_days=args.batch_interval_days)
        policy = BatchThompsonSampling(environment.variants, policy_seed=args.policy_seed)
        sends, batch_rows = run_campaign(
            environment, policy,
            total_sends=args.total_sends, batch_size=args.batch_size,
            control_share=args.control_share, assignment_seed=args.assignment_seed)
        rows = summarize(sends, policy, environment)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = args.output_dir.resolve() / f"field_{timestamp}"
        destination.mkdir(parents=True, exist_ok=False)
        for name, table in (("sends.csv", sends), ("batches.csv", batch_rows),
                            ("summary.csv", rows)):
            with (destination / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)

        hashes = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                  for n in ("claim_environment.py", "batch_thompson.py",
                            "run_field_simulation.py")}
        (destination / "settings.json").write_text(json.dumps({
            "simulation_only": True,
            "strategy": "batched_thompson_sampling_with_uniform_control",
            "assumed_rates": rates,
            "total_sends": args.total_sends,
            "batch_size": args.batch_size,
            "control_share": args.control_share,
            "window_days": args.window_days,
            "batch_interval_days": args.batch_interval_days,
            "maturation_lag_batches": environment.maturation_lag,
            "environment_seed": args.seed,
            "policy_seed": args.policy_seed,
            "assignment_seed": args.assignment_seed,
            "assignment_probabilities_logged": True,
            "python_version": platform.python_version(),
            "code_sha256": hashes,
            "created_utc": timestamp,
            "limitations": "Stationary assumed rates; no recipient features; "
                           "rates are assumptions, not measured values.",
        }, indent=2) + "\n", encoding="utf-8")

        total_conversions = sum(r["matured_conversions"] for r in rows)
        print("SIMULATED FIELD CAMPAIGN — ALL OUTCOMES SIMULATED")
        print(f"Assumed rates: {rates}")
        print(f"{args.total_sends:,} sends in batches of {args.batch_size:,}; "
              f"control share {args.control_share:.0%}; "
              f"maturation lag {environment.maturation_lag} batch(es)\n")
        print("Variant     Sent   Allocation   Conversions   Observed rate   Posterior mean")
        for row in rows:
            observed = "n/a" if row["observed_rate"] is None else f"{row['observed_rate']:.3%}"
            print(f"{row['variant']:7} {row['sent']:8,d} {row['allocation_share']:11.1%} "
                  f"{row['matured_conversions']:13,d} {observed:>15} "
                  f"{row['posterior_mean']:16.4%}")
        print(f"\nTotal conversions: {total_conversions:,}")
        print(f"Saved: {destination}")
        print("Single run. Use run_design_study.py for repeated runs with intervals.")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
