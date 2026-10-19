"""Run from the project root: python -m unittest discover -s pilot -p test_thompson_sampling.py

These tests check learning correctness, information boundaries, and integration.
They are implementation checks, not evidence of real-world policy improvement.
"""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from email_environment import EmailEnvironment
from thompson_sampling import ThompsonSampling, run_thompson


class ThompsonTests(unittest.TestCase):
    def test_exact_updates_and_no_unchosen_update(self):
        policy = ThompsonSampling(("A", "B"), alpha=2, beta=3)
        initial = policy.posterior()
        policy.select()
        self.assertEqual(initial, policy.posterior())  # Selection alone cannot learn.
        policy.update("B", 1)
        policy.update("B", 0)
        state = policy.posterior()
        self.assertEqual(state["A"], initial["A"])
        self.assertEqual(state["B"], {"alpha": 3.0, "beta": 4.0, "mean": 3 / 7})
        state["A"]["alpha"] = 999
        self.assertEqual(policy.posterior()["A"], initial["A"])  # Returned state is a copy.

    def test_invalid_pending_outcomes_and_priors(self):
        policy = ThompsonSampling(("A", "B"))
        before = policy.posterior()
        for value in (None, -1, 2, 0.5, "1", True):
            with self.assertRaises(ValueError):
                policy.update("A", value)
        with self.assertRaises(ValueError):
            policy.update("C", 1)
        self.assertEqual(before, policy.posterior())
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ThompsonSampling(("A", "B"), alpha=value)

    def test_reproducible_and_accounted(self):
        def once():
            return run_thompson(EmailEnvironment({"A": .01, "B": .013}, seed=42),
                                1000, policy_seed=123)
        first = once()
        self.assertEqual(first, once())
        counts, history, posterior = first
        self.assertEqual(sum(c["trials"] for c in counts.values()), 1000)
        self.assertEqual(history[-1]["cumulative_successes"], sum(r["reward"] for r in history))
        for arm, count in counts.items():
            self.assertEqual(posterior[arm]["alpha"], 1 + count["successes"])
            self.assertEqual(posterior[arm]["beta"], 1 + count["trials"] - count["successes"])

    def test_only_public_audience_interface_and_strong_signal(self):
        class Audience:
            # No probabilities or RNG are exposed: the policy must use this interface only.
            variants = ("A", "B", "C")
            def send(self, variant):
                return int(variant == "C")
        counts, _, _ = run_thompson(Audience(), 500, policy_seed=7)
        self.assertGreater(counts["C"]["trials"], 450)
        self.assertEqual(counts["A"]["successes"], 0)
        self.assertEqual(counts["B"]["successes"], 0)

    def test_cli_project_layout_outputs_and_modes(self):
        # A tiny fixture checks default path resolution without needing company data.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pilot = root / "pilot"
            pilot.mkdir()
            for name in ("email_environment.py", "thompson_sampling.py", "run_thompson.py"):
                shutil.copyfile(Path(__file__).with_name(name), pilot / name)
            data = root / "hubspot_email_campaign" / "hubspot_research_package"
            data.mkdir(parents=True)
            from email_environment import CAMPAIGN
            (data / "pilot_ab_baseline.csv").write_text(
                "campaign_name,A_delivered,A_clicks_7d,B_delivered,B_clicks_7d\n"
                f"{CAMPAIGN},10,0,10,10\n", encoding="utf-8")
            command = [sys.executable, str(pilot / "run_thompson.py")]
            def invoke(*args):
                return subprocess.run(command + list(args), cwd=temp, text=True, capture_output=True)
            self.assertEqual(invoke().returncode, 0)
            folder = next((pilot / "results").iterdir())
            with (folder / "history.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 20)
            settings = json.loads((folder / "settings.json").read_text())
            self.assertEqual(settings["mode"], "historical_click_calibration")
            self.assertFalse(settings["assignment_probabilities_logged"])
            # Three equal, zero-rate variants must yield no successes, including a one-trial run.
            self.assertEqual(invoke("--rates", "0", "0", "0", "--recipients", "1").returncode, 0)
            self.assertNotEqual(invoke("--rates", "0.01", "0.02", "--campaign", "x").returncode, 0)
            self.assertNotEqual(invoke("--rates", "-1", "0.02").returncode, 0)
            self.assertNotEqual(invoke("--prior-alpha", "0").returncode, 0)


if __name__ == "__main__":
    unittest.main()
