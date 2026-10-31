# AI-Augmented Optimization of Email Campaign Design

Master’s thesis at **BTU Cottbus–Senftenberg**, focused on optimizing **ProvenExpert’s business-profile claim invitations** using Bayesian learning.

## About the project

This project investigates whether **Thompson Sampling** can generate more profile claims than a fixed A/B split when conversions are rare, responses arrive late, and emails are sent in batches. It also explores whether simulation can help design a field experiment and predict its outcomes.

The main reward is a **completed profile claim within a defined observation window**, rather than an email open or click.

## What the code does

- Simulates invitation campaigns with configurable claim rates and response delays.
- Compares a uniform random control group with adaptive Thompson Sampling allocation.
- Updates the learning policy between batches, using only outcomes whose observation windows have closed.
- Records assignments, selection probabilities, batch summaries, and results.
- Runs repeated simulations to explore batch sizes, control shares, sending volumes, and differences between variants.

**The current code runs simulations only.** Conversion rates are assumptions, and no real emails are sent. The thesis describes how this framework can support a field experiment.

## Project files

| File or folder | Purpose |
|---|---|
| `pilot/claim_environment.py` | Simulated claims, delays, and observation windows |
| `pilot/batch_thompson.py` | Batched Thompson Sampling and assignment probabilities |
| `pilot/run_field_simulation.py` | One simulated campaign with control and adaptive groups |
| `pilot/run_design_study.py` | Repeated simulations for experiment design |
| `pilot/test_claim_pilot.py` | Tests for the environment, policy, and logging |
| `thesis/` | LaTeX dissertation, bibliography, and compiled PDF |

## Getting started

Use **Python 3.9 or later**. The campaign simulator uses the standard library; the design study also requires NumPy:

```bash
python -m pip install numpy
```

Run these commands from the project root:

```bash
# Simulate one campaign
python pilot/run_field_simulation.py

# Explore possible experiment designs
python pilot/run_design_study.py

# Run the tests
python -m unittest discover -s pilot -p "test_*.py"
```

Example with custom settings:

```bash
python pilot/run_field_simulation.py --rates 0.010 0.013 --total-sends 20000 --batch-size 2000 --control-share 0.40
```

Here, `--rates` specifies assumed conversion probabilities, and `--control-share 0.40` reserves 40% of each batch for uniform random allocation. Use `--help` on either runner to see more options.

## Results

Outputs are saved in timestamped folders under `pilot/results/`:

- **Campaign simulation:** `sends.csv`, `batches.csv`, `summary.csv`, and `settings.json`.
- **Design study:** `grid.csv` and `design_study.json`.

Saved settings and random seeds support reproducibility. Simulation results describe the chosen assumptions; they do not establish an improvement in a real campaign.
