"""Run from the project root:

    python -m unittest discover -s pilot -p "test_claim_pilot.py"

These are implementation checks: they verify that the code does what the thesis
says it does. They are not evidence about the real campaign.
"""

import math
import random
import unittest

from claim_environment import ClaimEnvironment
from batch_thompson import (BatchThompsonSampling, probability_greater,
                            logging_probability, uniform_probabilities)
from run_field_simulation import run_campaign, CONTROL, ADAPTIVE


class EnvironmentTests(unittest.TestCase):

    def test_pending_outcomes_are_withheld_not_failed(self):
        """An outcome whose window has not closed must not be visible at all."""
        environment = ClaimEnvironment({"A": 1.0, "B": 1.0}, seed=1)
        for _ in range(50):
            environment.send("A", batch_index=0)
        # Lag is one batch, so nothing is visible while allocating batch 0.
        self.assertEqual(environment.collect_matured(0), [])
        self.assertEqual(environment.pending_count(), 50)
        matured = environment.collect_matured(1)
        self.assertEqual(len(matured), 50)
        self.assertEqual(environment.pending_count(), 0)

    def test_maturation_lag_follows_window_and_interval(self):
        self.assertEqual(ClaimEnvironment(seed=1, window_days=7,
                                          batch_interval_days=7).maturation_lag, 1)
        self.assertEqual(ClaimEnvironment(seed=1, window_days=14,
                                          batch_interval_days=7).maturation_lag, 2)
        self.assertEqual(ClaimEnvironment(seed=1, window_days=7,
                                          batch_interval_days=1).maturation_lag, 7)

    def test_late_conversions_fall_outside_the_window(self):
        """A conversion arriving after the window is recorded as a non-conversion."""
        always = ClaimEnvironment({"A": 1.0}, seed=3, window_days=0,
                                  delay_weights=(0.5, 0.5))
        for _ in range(2000):
            always.send("A", 0)
        rate = sum(r for _, r in always.collect_matured(1)) / 2000
        self.assertAlmostEqual(rate, 0.5, delta=0.05)

    def test_rates_are_recovered_at_scale(self):
        environment = ClaimEnvironment({"A": 0.010, "B": 0.013}, seed=5)
        for index in range(200000):
            environment.send("A" if index % 2 == 0 else "B", 0)
        matured = environment.collect_matured(1)
        for variant, expected in (("A", 0.0093), ("B", 0.0121)):
            rows = [r for v, r in matured if v == variant]
            self.assertAlmostEqual(sum(rows) / len(rows), expected, delta=0.0015)

    def test_same_seed_gives_identical_recipient_randomness(self):
        """Common random numbers: the draw sequence must not depend on the actions."""
        def rewards(sequence):
            environment = ClaimEnvironment({"A": 0.5, "B": 0.5}, seed=11)
            for index, variant in enumerate(sequence):
                environment.send(variant, 0)
            return [r for _, r in environment.collect_matured(1)]
        self.assertEqual(rewards("AAAA" * 50), rewards("ABAB" * 50))

    def test_rejects_invalid_configuration(self):
        for bad in ({"A": 1.5}, {"A": -0.1}, {"A": float("nan")}, {"": 0.1}, {}):
            with self.assertRaises(ValueError):
                ClaimEnvironment(bad, seed=1)
        with self.assertRaises(ValueError):
            ClaimEnvironment(seed=1, window_days=-1)
        with self.assertRaises(ValueError):
            ClaimEnvironment(seed=1).send("Z", 0)


class PolicyTests(unittest.TestCase):

    def test_closed_form_matches_monte_carlo(self):
        rng = random.Random(0)
        for a1, b1, a2, b2 in ((1, 1, 1, 1), (3, 47, 8, 42), (21, 1979, 34, 1966)):
            draws = 200000
            mc = sum(1 for _ in range(draws)
                     if rng.betavariate(a1, b1) > rng.betavariate(a2, b2)) / draws
            self.assertAlmostEqual(probability_greater(a1, b1, a2, b2), mc, delta=0.005)

    def test_selection_probabilities_are_a_distribution(self):
        for variants in (("A", "B"), ("A", "B", "C")):
            policy = BatchThompsonSampling(variants, propensity_samples=5000)
            probabilities = policy.selection_probabilities()
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
            for value in probabilities.values():
                self.assertGreaterEqual(value, 0.0)

    def test_posterior_updates_only_the_chosen_variant(self):
        policy = BatchThompsonSampling(("A", "B"), alpha=2, beta=3)
        before = policy.posterior()
        policy.select()
        self.assertEqual(before, policy.posterior())   # selecting cannot learn
        policy.update("B", 1)
        policy.update("B", 0)
        after = policy.posterior()
        self.assertEqual(after["A"], before["A"])
        self.assertEqual(after["B"], {"alpha": 3.0, "beta": 4.0, "mean": 3 / 7})

    def test_posterior_is_frozen_within_a_batch(self):
        policy = BatchThompsonSampling(("A", "B"))
        frozen = policy.begin_batch()
        policy.update("A", 1)
        for _ in range(20):
            policy.update("A", 1)
        # Updating must not change the probabilities in force for the current batch.
        self.assertEqual(frozen, policy.frozen_probabilities())
        # ... but it must change them at the next boundary.
        self.assertNotEqual(frozen, policy.begin_batch())

    def test_rejects_invalid_rewards_and_priors(self):
        policy = BatchThompsonSampling(("A", "B"))
        for value in (None, -1, 2, 0.5, "1", True):
            with self.assertRaises(ValueError):
                policy.update("A", value)
        with self.assertRaises(ValueError):
            policy.update("Z", 1)
        for value in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                BatchThompsonSampling(("A", "B"), alpha=value)

    def test_strong_signal_is_found(self):
        class Audience:
            variants = ("A", "B", "C")
            def send(self, variant):
                return int(variant == "C")
        policy = BatchThompsonSampling(Audience.variants, policy_seed=7,
                                       propensity_samples=2000)
        for _ in range(400):
            chosen = policy.select()
            policy.update(chosen, int(chosen == "C"))
        self.assertGreater(policy.posterior()["C"]["mean"], 0.8)


class LoggingTests(unittest.TestCase):

    def test_logging_probability_mixes_both_streams(self):
        control = uniform_probabilities(("A", "B"))
        policy = {"A": 0.9, "B": 0.1}
        self.assertAlmostEqual(
            logging_probability("A", 0.4, control, policy), 0.4 * 0.5 + 0.6 * 0.9)
        total = sum(logging_probability(v, 0.4, control, policy) for v in ("A", "B"))
        self.assertAlmostEqual(total, 1.0)

    def test_every_send_carries_a_usable_propensity(self):
        """Off-policy evaluation needs a strictly positive probability on every send."""
        environment = ClaimEnvironment({"A": 0.010, "B": 0.013}, seed=2)
        policy = BatchThompsonSampling(environment.variants, policy_seed=9)
        sends, batches = run_campaign(environment, policy, total_sends=6000,
                                      batch_size=2000, control_share=0.40)
        self.assertEqual(len(sends), 6000)
        self.assertEqual(len(batches), 3)
        for send in sends:
            self.assertGreater(send["assignment_probability"], 0.0)
            self.assertLessEqual(send["assignment_probability"], 1.0)
        # The control share bounds the propensity away from zero, which is what
        # keeps the inverse-propensity estimator's variance finite.
        self.assertGreaterEqual(min(s["assignment_probability"] for s in sends),
                                0.40 / 2 - 1e-9)

    def test_control_share_is_respected_in_every_batch(self):
        environment = ClaimEnvironment({"A": 0.010, "B": 0.013}, seed=4)
        policy = BatchThompsonSampling(environment.variants, policy_seed=4)
        sends, _ = run_campaign(environment, policy, total_sends=8000,
                                batch_size=2000, control_share=0.25)
        for batch in range(4):
            rows = [s for s in sends if s["batch"] == batch]
            self.assertEqual(sum(1 for s in rows if s["stream"] == CONTROL), 500)
            self.assertEqual(sum(1 for s in rows if s["stream"] == ADAPTIVE), 1500)

    def test_control_stream_allocation_stays_uniform(self):
        """The control stream must not drift with the adaptive stream."""
        environment = ClaimEnvironment({"A": 0.01, "B": 0.20}, seed=6)
        policy = BatchThompsonSampling(environment.variants, policy_seed=6)
        sends, _ = run_campaign(environment, policy, total_sends=20000,
                                batch_size=2000, control_share=0.50)
        control = [s for s in sends if s["stream"] == CONTROL]
        share_b = sum(1 for s in control if s["variant"] == "B") / len(control)
        self.assertAlmostEqual(share_b, 0.5, delta=0.03)
        # Meanwhile the adaptive stream should have moved decisively to B.
        adaptive = [s for s in sends if s["stream"] == ADAPTIVE]
        self.assertGreater(sum(1 for s in adaptive if s["variant"] == "B") / len(adaptive),
                           0.9)

    def test_run_is_reproducible(self):
        def once():
            environment = ClaimEnvironment({"A": 0.010, "B": 0.013}, seed=8)
            policy = BatchThompsonSampling(environment.variants, policy_seed=8)
            return run_campaign(environment, policy, total_sends=4000,
                                batch_size=1000, control_share=0.30)
        self.assertEqual(once(), once())


if __name__ == "__main__":
    unittest.main()
