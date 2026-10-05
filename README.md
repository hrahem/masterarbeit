# Email campaign pilot: simulated A/B baseline

This guide explains the two current Python components, how to run them, and how to discuss them at the thesis defense. Python 3.9 or later is required. Both files use only Python's standard library; no `pip install` is needed.

## 1. What has been implemented?

| Component | Responsibility |
|---|---|
| `email_environment.py` | Simulates a click or no-click response for a selected email version. Also loads campaign rates from the prepared CSV. |
| `run_ab_test.py` | Randomly assigns approximately half of the simulated deliveries to A and half to B, obtains responses, and saves results. |
| `pilot_ab_baseline.csv` | Supplies historical counts for the selected campaign. This is input data, not code. |

The environment answers **“Did this simulated recipient click?”** The policy answers **“Which version should this recipient receive?”** Keeping these responsibilities separate allows a different policy, such as Thompson Sampling, to use the same response-generating mechanism later.

The current implementation is a stationary, noncontextual Bernoulli simulation. “Stationary” means the click probabilities remain constant during a run. “Noncontextual” means there are no individual recipient features. “Bernoulli” means each response is either 1 (click) or 0 (no click).

No real email is sent. There is no HubSpot connection, HTML processing, predictive-model training, personalization, conversion modeling, or Thompson Sampling implementation in these two files yet.

## 2. Where to place the files

Use these paths relative to the thesis project root:

| Path | Purpose |
|---|---|
| `pilot/email_environment.py` | Simulated audience module |
| `pilot/run_ab_test.py` | Experiment runner |
| `pilot/README.md` | This guide |
| `hubspot_email_campaign/hubspot_research_package/pilot_ab_baseline.csv` | Existing prepared input |
| `pilot/results/` | Created automatically to hold simulation outputs |
| `thesis/` | The LaTeX dissertation |

The runner uses `Path(__file__).resolve().parent.parent` to locate the project root. `__file__` means the runner's own filename, `resolve()` makes its path absolute, and two `.parent` operations move up from the file to `pilot`, then to the project root.

The default input and output paths therefore do not depend on where your terminal is open. However, a relative script path or an explicitly supplied relative argument is interpreted from the terminal's current directory. The commands below assume you opened the terminal at the project root.

## 3. Run the A/B baseline

```bash
python pilot/run_ab_test.py
```

On Windows, if `python` is unavailable but the Python launcher is installed, use `py` instead.

The default run selects `DE_NL-CW12-25_DACH-DE-PAIDvA`, reads its counts, computes probabilities, and simulates 5,627 deliveries. It creates the complete balanced assignment before observing any rewards. With an odd budget, one variant receives one extra delivery; the policy randomly chooses which variant gets it.

With the supplied input and default seeds, the verified output is:

| Version | Simulated deliveries | Simulated clicks | Simulated click rate |
|---|---:|---:|---:|
| A | 2,814 | 44 | 1.56% |
| B | 2,813 | 64 | 2.28% |
| Total | 5,627 | 108 | 1.92% |

These are generated outcomes. The historical counts are A: 45 clicks / 2,815 deliveries and B: 73 clicks / 2,812 deliveries. Simulated counts need not reproduce historical counts. The simulation balances assignments itself rather than replaying historical assignments.

## 4. Why this campaign was selected

The paid newsletter is a practical first debugging scenario because it has the largest cleaned delivery budget among the seven rows in the prepared pilot CSV, clicks in both variants, and a visible empirical difference between A and B. The original audit identified a straightforward 50/50 test, avoiding the interpretation of a later winner-rollout phase for this starting example.

Its cleaned pilot probabilities are:

- A: `45 / 2815`, approximately 1.60%.
- B: `73 / 2812`, approximately 2.60%.

These are estimates from observed data, not known real-world response probabilities. The difference makes it useful for checking whether a future learning policy can discover the higher-rate action in a constructed scenario. It is not a representative or independently selected evaluation benchmark. Selecting only a campaign with a clear gap could make the final comparison overly favorable to adaptive allocation.

For thesis evaluation, specify campaign inclusion criteria in advance and report results across suitable campaigns, including near-equal rates and scenarios where A has the higher rate.

## 5. Select another campaign

The runner selects **one campaign per invocation**, but supports every valid row in the input CSV. The campaign name must match `campaign_name` exactly, including capitalization.

```bash
python pilot/run_ab_test.py --campaign "NL_Webinar_DACH-DE-LEADs_A"
```

The selected campaign changes both the click probabilities and the default recipient budget. `--recipients` overrides only the budget.

The current prepared CSV contains:

| Campaign name | A deliveries / clicks | B deliveries / clicks | Default budget |
|---|---:|---:|---:|
| `DACH_DE_BASIC_checkoutfeb25_2A` | 198 / 4 | 195 / 2 | 393 |
| `DE_NL-CW12-25_DACH-DE-LEADSvA` | 1,259 / 22 | 1,245 / 24 | 2,504 |
| `DE_NL-CW12-25_DACH-DE-PAIDvA` | 2,815 / 45 | 2,812 / 73 | 5,627 |
| `DE_NL-CW12-25_ROW-DE-PAIDvA` | 93 / 6 | 84 / 1 | 177 |
| `NL_Webinar_DACH-DE-LEADs_A` | 1,175 / 29 | 1,200 / 30 | 2,375 |
| `dach_de_plus_m_consistency26m_2` | 60 / 0 | 58 / 0 | 118 |
| `dach_de_plus_m_consistency26m_3` | 61 / 1 | 56 / 0 | 117 |

Clicks in this table use the prepared seven-day outcome definition. Eligibility and cleaning were performed upstream; the loader does not repeat those audits. The input is already a selected subset of the original exports.

Zero observed clicks become exactly zero probability in the current code. Thus the consistency26m_2 scenario always generates zero clicks for both variants. This does not establish that real people would never click. Sparse campaigns need explicit uncertainty or smoothing sensitivity analyses before substantive conclusions. The current implementation intentionally uses raw empirical ratios without smoothing.

## 6. What the two seeds mean

A seed is an integer that initializes a pseudo-random generator. It makes a sequence reproducible. It is not an audience size, click probability, or quality setting. The numbers 42 and 123 are arbitrary defaults.

| Argument | Controls | Does not change |
|---|---|---|
| `--seed` | Uniform random numbers used to generate click responses | The probabilities loaded from the CSV |
| `--policy-seed` | Assignment shuffle and which variant gets an odd extra delivery | The response generator's random-number sequence |

**Environment example:** suppose the next uniform draw is `0.020`. Under the default A probability of about `0.016`, the response is 0. Under B's probability of about `0.026`, it would be 1. The code generates only the chosen version's reward; the alternative is an explanation of the mechanism.

**Allocation example:** one policy seed might give an order beginning `A, B, B, A`; another might give `B, A, B, A`. Both remain balanced overall. These short sequences are illustrative, not the literal outputs of particular seeds.

```bash
python pilot/run_ab_test.py --seed 42 --policy-seed 123
python pilot/run_ab_test.py --seed 43 --policy-seed 123
python pilot/run_ab_test.py --seed 42 --policy-seed 124
```

The second command changes response randomness while keeping the assignment sequence fixed for the same budget. The third changes allocation randomness while preserving the sequence of uniform response draws. Changing allocation can still change observed clicks: a draw is compared with the probability of whichever variant is selected at that step.

The same inputs, seeds, code, and compatible Python runtime reproduce simulation outcomes. Output timestamps differ between executions, so entire output directories are not byte-for-byte identical. Preserve the Python version as well for long-term reproducibility; the current settings file does not record it automatically. Different seeds can occasionally produce identical totals by chance.

Separate generators prevent the policy's shuffle or future Bayesian sampling from consuming random numbers that would otherwise generate audience responses.

For a future paired A/B-versus-Thompson comparison, create a fresh environment for each policy with the same environment seed and budget within each repetition. Change that seed between repetitions and give policies their own generators. Sharing the uniform sequence is a simulation design choice, not evidence that real recipients' unobserved responses are known. Equal policy seeds across different algorithms do not imply equal decisions or random-number consumption.

## 7. All runner arguments

| Argument | Default | Meaning |
|---|---|---|
| `--campaign` | Paid newsletter named above | Exact campaign to load |
| `--baseline-csv` | Prepared CSV in your data folder | Alternative input CSV path |
| `--seed` | `42` | Audience response seed |
| `--policy-seed` | `123` | Allocation seed |
| `--recipients` | A deliveries + B deliveries | Override simulated budget; at least 2 |
| `--output-dir` | `pilot/results` | Parent folder for a new run directory |
| `--help` | — | Show command-line usage |

Example with multiple options:

```bash
python pilot/run_ab_test.py --campaign "NL_Webinar_DACH-DE-LEADs_A" --seed 43 --policy-seed 124 --recipients 2000
```

Example with a custom CSV location:

```bash
python pilot/run_ab_test.py --baseline-csv "hubspot_email_campaign/hubspot_research_package/pilot_ab_baseline.csv"
```

There is no `--all-campaigns` or `--list-campaigns` argument in this version. Use the names above or inspect the CSV, and run each campaign separately.

## 8. Running the environment by itself

```bash
python pilot/email_environment.py
```

This runs two separate diagnostic scenarios: all A and all B. It does not run the 50/50 baseline or save the runner's result files. Without an input CSV, this diagnostic uses built-in counts for the default campaign.

To change the diagnostic campaign, supply both options:

```bash
python pilot/email_environment.py --baseline-csv "hubspot_email_campaign/hubspot_research_package/pilot_ab_baseline.csv" --campaign "NL_Webinar_DACH-DE-LEADs_A"
```

The diagnostic supports `--seed`, `--recipients`, `--baseline-csv`, and `--campaign`. It has no `--policy-seed` because there is no randomized A/B allocator in the diagnostic. You do not need to run it before `run_ab_test.py`: importing the module makes its class and loader available automatically.

## 9. Read the saved results

Each invocation creates a new timestamped subfolder under `pilot/results`.

| File | Contents |
|---|---|
| `history.csv` | `step`, `variant`, `click`, `cumulative_clicks`; one row per simulated delivery |
| `summary.csv` | `variant`, `deliveries`, `clicks`, `click_rate`; A, B, and TOTAL rows |
| `settings.json` | Campaign, input path, source/code hashes, assumed rates, budgets, seeds, timestamp, assumptions |

A CSV click rate of `0.02` means 2%. The terminal displays percentages automatically. The total rate is total clicks divided by total deliveries, not an unweighted mean of the two displayed percentages.

SHA-256 hashes fingerprint file contents so later changes can be detected. They do not encrypt or anonymize data. Adding comments changes a code hash even though the simulation behavior is unchanged.

## 10. Code walkthrough and Python notation

The annotated files explain each executable operation and related multiline expression. Read them in this order:

1. `load_campaign_rates`: select the exact CSV row, validate counts, calculate `clicks / delivered`, and sum deliveries.
2. `EmailEnvironment.__init__`: validate inputs and store a private generator, probabilities, budget, and round counter.
3. `EmailEnvironment.send`: validate the action and remaining budget, draw `u`, return `int(u < p)`, increment the counter.
4. `run_ab`: construct a balanced assignment, shuffle it once, execute every delivery, update counts, and record history.
5. `main`: parse arguments, load data, create the environment, call the policy, save files, and print results.
6. `demo`: an independent diagnostic used only when executing the environment file directly.

| Python notation | Meaning here |
|---|---|
| `self` | The particular audience instance whose state is being accessed |
| `_probabilities`, `_rng` | Internal attributes by naming convention; Python does not enforce their privacy |
| `None` | No optional value supplied |
| `Optional[int]` | Type hint meaning integer or None; hints alone do not validate runtime inputs |
| `Mapping[str, float]` | Type hint for a dictionary-like mapping from string labels to numeric rates |
| `*` in the constructor | Following arguments must be named, such as `seed=42` |
| `@property` | Allows a method to be read like an attribute: `audience.variants` |
| `//`, `%` | Integer division and remainder, used to balance odd/even budgets |
| `+=` | Add to the current value |
| `int(condition)` | Convert True to 1 and False to 0 |
| `**counts[v]` | Copy a dictionary's entries into another dictionary |
| List/dictionary comprehension | Build a collection by iterating in a compact expression |
| `enumerate(..., start=1)` | Pair each assignment with a delivery number starting at 1 |
| `with ... as f` | Open a file and close it automatically after the block |
| `try` / `except` | Handle expected input and filesystem errors cleanly |
| `f"...{value}..."` | Insert a value into a string |
| `:10.2%` | Display a percentage with two decimals in a field at least ten characters wide |
| `:20,d` | Display an integer with thousands separators and minimum field width 20 |
| `\n` | Newline character |
| `if __name__ == "__main__"` | Run the entry point only when executing the file directly |

Closing brackets and continuation lines complete the expression above them; they do not represent additional simulation steps.

## 11. Questions you may receive at your defense

**What is your current contribution?**
A reproducible experiment scaffold separating response generation from allocation, calibrated to cleaned company campaign counts. The baseline is an established method; the later contribution is adaptation and evaluation under explicit email-campaign assumptions.

**Is this already an AI model trained on recipient data?**
No. The current environment uses one empirical click rate per variant. Thompson Sampling will learn an allocation policy from simulated responses. A recipient-level predictive simulator would be a separate component requiring suitable features and validation.

**Why does `u < p` simulate a click?**
A uniform random number falls in the interval from 0 to p with probability p. The comparison therefore produces a binary outcome whose probability of 1 is p.

**Does the baseline learn?**
No. It fixes the balanced assignment before seeing responses. It measures cumulative clicks under fixed allocation; it does not implement a significance test, early stopping, or winner rollout.

**Does a higher simulated B rate prove that B is better in reality?**
No. B's larger assumed probability was supplied to this scenario using historical counts. Generated outcomes assess policies under those assumptions; they do not independently validate the historical difference.

**Why might the simulated rate differ from the input rate?**
Finite random samples fluctuate. The input probability is an expected frequency, not a quota enforced by the code.

**Why not simulate millions of deliveries and call that a larger dataset?**
More repetitions can reduce Monte Carlo error when estimating performance in the assumed environment. They cannot remove uncertainty in the historical probabilities or create new evidence about real recipients. A larger within-run budget also changes the decision problem by giving the policy more time to learn.

**Is feedback timing realistic?**
Not yet. A reward represents whether a click occurred within seven days, but is revealed immediately in simulation. A more realistic comparison must model feedback delays and avoid treating pending outcomes as failures.

**Can you claim causal improvement or generalization?**
Not from this pilot. It is an empirical-rate simulation, not held-out prediction, validated counterfactual replay, or a live randomized policy comparison. Historical assignment mechanisms and uncertainty require separate consideration. Some campaigns also share content families and should not be treated as fully independent evidence.

## 12. Next implementation steps

1. Run the default campaign and the webinar campaign to understand loading, outputs, and seed behavior.
2. Implement Beta-Bernoulli Thompson Sampling with its own policy generator. It should observe only chosen actions' rewards and must not read the environment's probabilities.
3. Build a repeated paired comparison using fresh environments and a predefined sequence of seeds. Report average total clicks, paired differences, and variability across runs.
4. Evaluate suitable campaigns and synthetic sensitivity scenarios, including equal rates, small gaps, sparse clicks, and uncertainty about the empirical rates. Separate Monte Carlo variability from historical-estimation uncertainty.
5. Add delayed feedback and realistic sending batches before interpreting the experiment as an operational email allocation study.
6. Describe architecture, data preparation, and assumptions in Chapter 5; report results and limitations in Chapter 6. Recipient-context models and sequential drip RL require additional data and validation beyond this baseline.

## 13. Troubleshooting

- **Cannot import `email_environment`:** keep both Python files in the same `pilot` directory.
- **CSV not found:** check the documented folder structure or provide `--baseline-csv` explicitly.
- **Expected exactly one row:** copy the exact campaign name from the CSV and ensure it appears once.
- **Missing CSV column:** use the prepared baseline summary, not a raw recipients export.
- **All zero clicks in a tiny campaign:** inspect historical counts; zero empirical rates always generate zeros in this version.
- **Results identical on rerun:** expected with the same seeds and inputs; change seeds for another realization.
- **The output differs after modifying inputs/code/runtime:** preserve those inputs and versions when documenting reproducibility.
