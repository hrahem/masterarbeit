"""Task 2: fixed balanced A/B simulation. Python 3.9+, standard library only.

Place beside email_environment.py in your project's pilot folder, then run:
    python pilot/run_ab_test.py

This is a simulation conditional on delivery, NOT a live experiment or a
statistical significance test. The CSV's empirical rates define the environment;
they are not known population probabilities. Seven-day outcomes are revealed
immediately in this initial benchmark. No HubSpot connection or real sends.
"""

# Read command-line options such as --campaign and --seed.
import argparse
# Read and write comma-separated tables using column names.
import csv
# Calculate SHA-256 file fingerprints to record exactly which data and code were used.
import hashlib
# Write structured experiment settings in a human-readable format.
import json
# Generate repeatable pseudo-random numbers; no external package is needed.
import random
# Create a UTC timestamp for each result directory.
from datetime import datetime, timezone
# Represent file paths portably on Windows, macOS, and Linux.
from pathlib import Path

# Reuse the default name, audience class, and CSV loader from the neighboring file.
from email_environment import CAMPAIGN, EmailEnvironment, load_campaign_rates


# __file__ is this script; two parents go from pilot/run_ab_test.py to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Locate the prepared CSV using your project structure, independent of the terminal folder.
DEFAULT_BASELINE = (
    # The / operator joins path components portably; here it is not numeric division.
    PROJECT_ROOT / "hubspot_email_campaign" / "hubspot_research_package"
    # Append the CSV filename to the data directory.
    / "pilot_ab_baseline.csv"
)


# The fixed-allocation policy accepts an audience, a delivery budget, and its own seed.
def run_ab(audience, recipients, policy_seed=123):
    """Choose the complete balanced allocation before observing any rewards.

    With an odd budget, randomly choose which variant gets the extra recipient.
    Shuffle using a policy RNG separate from the environment's reward RNG.
    The policy sees only available variants and chosen-action outcomes.
    """
    # Need an integer budget of at least two so both variants receive a delivery.
    if isinstance(recipients, bool) or not isinstance(recipients, int) or recipients < 2:
        # Explain why this budget cannot support the baseline.
        raise ValueError("Use at least two simulated recipients.")
    # Require exactly the two supported action labels; set ignores their order.
    if set(audience.variants) != {"A", "B"}:
        # Reject an environment with missing or extra variants.
        raise ValueError("This baseline requires variants A and B.")
    # Create a dedicated allocation generator; it cannot consume environment randomness.
    rng = random.Random(policy_seed)
    # // is integer division; make equally sized lists, then concatenate them with +.
    allocation = ["A"] * (recipients // 2) + ["B"] * (recipients // 2)
    # % computes a remainder; an odd budget leaves one recipient unassigned.
    if recipients % 2:
        # Randomly give that extra delivery to A or B so neither is systematically favored.
        allocation.append(rng.choice(("A", "B")))
    # Randomly reorder the complete allocation in place before seeing any clicks.
    rng.shuffle(allocation)
    # A dictionary comprehension creates independent delivery and click counters for each arm.
    counts = {v: {"deliveries": 0, "clicks": 0} for v in ("A", "B")}
    # Start an empty list that will hold one record per simulated delivery.
    history = []
    # Track total clicks observed so far across both variants.
    cumulative_clicks = 0
    # Walk through assignments; enumerate gives each delivery a number starting at one.
    for step, variant in enumerate(allocation, start=1):
        # Ask for the chosen variant's response: 1 for click or 0 for no click.
        click = audience.send(variant)
        # Add one to the selected variant's delivery counter.
        counts[variant]["deliveries"] += 1
        # Add the binary reward; a no-click adds zero.
        counts[variant]["clicks"] += click
        # Update the combined running click count.
        cumulative_clicks += click
        # Save an auditable record of the choice and observed outcome.
        history.append({
            # Store the delivery index, selected variant, and binary click outcome.
            "step": step, "variant": variant, "click": click,
            # Store the total clicks up to and including this delivery.
            "cumulative_clicks": cumulative_clicks,
        })
    # Return summary counters and the full sequence; the caller decides how to save them.
    return counts, history


# Coordinate command-line parsing, data loading, simulation, reporting, and saving.
def main():
    # __doc__ is the opening module description, reused in --help.
    parser = argparse.ArgumentParser(description=__doc__)
    # Accept a CSV path; type=Path converts the supplied text to a Path object.
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    # Select an exact campaign_name; use CAMPAIGN when this option is omitted.
    parser.add_argument("--campaign", default=CAMPAIGN)
    # The environment seed controls random click responses, not the click probabilities.
    parser.add_argument("--seed", type=int, default=42, help="Audience randomness.")
    # This seed controls the assignment shuffle and odd extra recipient, not responses.
    parser.add_argument("--policy-seed", type=int, default=123, help="Allocation randomness.")
    # Optionally override the simulated audience size; this does not add real evidence.
    parser.add_argument("--recipients", type=int, help="Default: campaign delivery count.")
    # Choose the parent output directory; each invocation creates a new subdirectory.
    parser.add_argument("--output-dir", type=Path,
                        # Default output location is results beside this script, inside pilot.
                        default=Path(__file__).resolve().parent / "results")
    # Parse terminal arguments; access each option as an attribute such as args.seed.
    args = parser.parse_args()
    # Run the main operation; report expected input or filesystem errors below.
    try:
        # Convert the input path into an absolute path for reading and recording.
        baseline = args.baseline_csv.resolve()
        # Fail clearly when the expected CSV is missing or points to a directory.
        if not baseline.is_file():
            # Construct a readable error explaining the missing input.
            raise ValueError(
                f"Campaign CSV not found: {baseline}\n"
                "Check the folder structure or supply --baseline-csv PATH."
            )
        # Load the selected row; changing campaign changes both rates and the default budget.
        probabilities, campaign_budget = load_campaign_rates(baseline, args.campaign)
        # Use the campaign size unless --recipients explicitly overrides it.
        budget = campaign_budget if args.recipients is None else args.recipients
        # Create fresh simulated responses with the selected rates, response seed, and budget.
        audience = EmailEnvironment(probabilities, seed=args.seed, horizon=budget)
        # Run the fixed A/B policy using a different generator for its assignments.
        counts, history = run_ab(audience, budget, args.policy_seed)
        # Sum the two variant click counters.
        total_clicks = sum(row["clicks"] for row in counts.values())

        # A new directory preserves each previous run, including identical-seed runs.
        # Format UTC as a filename-safe timestamp with microseconds; Z denotes UTC.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        # Put this run inside a newly named subdirectory of the chosen output directory.
        destination = args.output_dir.resolve() / f"ab_{timestamp}"
        # Create missing parents; refuse a name collision rather than overwrite previous results.
        destination.mkdir(parents=True, exist_ok=False)
        # Write a UTF-8 CSV; newline="" avoids extra blank lines on Windows.
        with (destination / "history.csv").open("w", newline="", encoding="utf-8") as f:
            # Use the first history record's keys as the column order.
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            # Write the CSV column names once.
            writer.writeheader()
            # Write all delivery records as CSV rows.
            writer.writerows(history)
        # Build one summary row per variant using a list comprehension.
        rows = [
            # ** unpacks this variant's delivery and click counters into the new dictionary.
            {"variant": v, **counts[v],
             # Compute clicks/deliveries; stored as a fraction such as 0.02, meaning 2%.
             "click_rate": counts[v]["clicks"] / counts[v]["deliveries"]}
            # Produce summary rows in A-then-B order.
            for v in ("A", "B")
        ]
        # Add the combined summary row; its rate uses the combined denominator.
        rows.append({"variant": "TOTAL", "deliveries": budget,
                     "clicks": total_clicks, "click_rate": total_clicks / budget})
        # Open a separate CSV for the compact per-variant and total results.
        with (destination / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            # Use the summary dictionary keys as CSV column names.
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            # Write the CSV column names once.
            writer.writeheader()
            # Save the A, B, and TOTAL summary rows.
            writer.writerows(rows)
        # Record inputs and assumptions for reproducibility; these values are not policy inputs.
        settings = {
            # Flag the results explicitly as synthetic.
            "simulation_only": True,
            # Name the allocation method implemented by this runner.
            "strategy": "fixed_balanced_randomized_ab",
            # Record the exact selected campaign name.
            "campaign": args.campaign,
            # Record the source CSV's absolute path.
            "baseline_csv": str(baseline),
            # Fingerprint the source bytes; this is provenance, not encryption or anonymization.
            "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            # Fingerprint the response-generating code, including its comments.
            "environment_sha256": hashlib.sha256(
                # Locate the neighboring environment file, read its bytes, and format its digest as hex.
                Path(__file__).with_name("email_environment.py").read_bytes()).hexdigest(),
            # Fingerprint this runner so later code changes can be detected.
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            # Record empirical rates defining this scenario; the policy never reads these settings.
            "assumed_probabilities": probabilities,
            # Keep the original historical budget even when the simulated budget is overridden.
            "historical_campaign_budget": campaign_budget,
            # Record the actual number of simulated deliveries requested.
            "simulated_deliveries": budget,
            # Record the seed controlling response randomness.
            "environment_seed": args.seed,
            # Record the independent seed controlling allocation randomness.
            "policy_seed": args.policy_seed,
            # Record the output creation time; timestamps differ even for repeated identical simulations.
            "created_utc": timestamp,
            # Document the simplifying assumption of instantly revealed seven-day outcomes.
            "feedback": "immediate revelation of completed seven-day outcome",
            # Record limits so these outputs are not mistaken for live experimental evidence.
            "limitations": "Stationary aggregate rates; no personalization or delay. "
                           "No significance test or real-world improvement claim.",
        }
        # Save settings to JSON alongside the two result tables.
        (destination / "settings.json").write_text(
            # Serialize with two-space indentation and a final newline for readability.
            json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print("FIXED 50/50 A/B SIMULATION")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print(f"Campaign: {args.campaign}")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print(f"Audience seed: {args.seed}; allocation seed: {args.policy_seed}")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print("\nVariant   Simulated deliveries   Simulated clicks   Click rate")
        # Display each summary row in the terminal.
        for row in rows:
            # Display diagnostic or result text; f-strings insert values and format percentages.
            print(f"{row['variant']:7}   {row['deliveries']:20,d}   "
                  f"{row['clicks']:16,d}   {row['click_rate']:10.2%}")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print(f"\nResults saved to: {destination}")
        # Display diagnostic or result text; f-strings insert values and format percentages.
        print("One simulated run; do not interpret it as evidence of a real-world winner.")
    # Catch file errors, invalid values, and missing CSV columns.
    except (OSError, ValueError, KeyError) as exc:
        # Print a readable error and usage instructions, then exit unsuccessfully.
        parser.error(str(exc))


# Execute the entry point only when run directly; importing the file does not run it.
if __name__ == "__main__":
    # Start the runner when this file is executed directly.
    main()
