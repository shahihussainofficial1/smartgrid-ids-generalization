# Do Smart Grid Intrusion Detectors Learn Attacks or Operating Conditions?

Evaluation of machine learning attack detectors for power systems on (1) the public MSU/ORNL power system attack dataset recorded on a PMU-relay testbed and (2) a 42-day IEEE 14-bus simulation with phasor measurement units (PMUs), weighted least-squares state estimation and false data injection (FDI).

The preprint is available in [`paper/main.pdf`](paper/main.pdf).

## Key findings

**PMU-relay testbed (binary task: attack vs. natural event / normal operation, random forest)**

| Evaluation | Macro-F1 | Accuracy |
|---|---|---|
| Random 70/30 split of samples | 0.90 | 91.7% |
| Leave-one-set-out (15 sets) | 0.61 | 68.9% (always predicting "attack": 71.0%) |

- The 15 sets can be told apart with 95.5% accuracy from normal-operation samples alone, so a random split rewards recognition of the operating condition.
- Load-independent physics features (sequence-component ratios, phase unbalance, line-end consistency) shrink the gap between the two protocols from 0.29 to 0.04 but do not raise the grouped score above 0.63.

**IEEE 14-bus simulation (stealthy FDI, alarm threshold calibrated to 1% false alarms on held-out normal data)**

| Detector | Future days | Unseen target buses |
|---|---|---|
| Chi-square bad data detector | 1.4% | 1.4% |
| Random forest, PMU measurements + residual | 99.9% (FAR 1.4%) | 86% (FAR 1.8%) |
| Random forest trained on the full training period, default threshold of 0.5 | 95% | 62% (ROC-AUC 0.996) |

## System diagrams

The paper contains a single-line and communication diagram of the testbed (`paper/figures/testbed.tex`), the IEEE 14-bus system with PMU placement (`paper/figures/ieee14.tex`) and the simulated detection chain (`paper/figures/pipeline.tex`), all drawn in TikZ.

## Repository structure

```
scripts/download_data.py   download and extract the MSU/ORNL power system attack datasets
src/data.py                load the 15 sets, binary / three-class / scenario labels
src/features.py            physics features and per-set normalization
src/run_real.py            random split vs. leave-one-set-out on the testbed data
src/simulate.py            IEEE 14-bus time-series power flow, PMUs, WLS state estimation, FDI attacks
src/run_sim.py             BDD, max.-deviation detector and classifiers on the simulation
src/make_figures.py        result figures and LaTeX tables
results/                   numeric results used in the paper
paper/                     preprint in IEEE conference format (main.pdf, main.tex, references.bib, figures)
```

## Reproducing the results

Requires Python 3.10+ and runs on a CPU. The simulation takes about 20 minutes and the testbed experiments about 2 hours on a laptop.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
cd src
python data.py
python run_real.py
python simulate.py
python run_sim.py
python make_figures.py
```

## Data

U. Adhikari, S. Pan, T. Morris, R. Borges and J. Beaver, *Power system attack datasets*, Mississippi State University and Oak Ridge National Laboratory, 2014. The data are not redistributed here; `scripts/download_data.py` downloads them from the providers.

## Citation

```bibtex
@misc{shahi2026smartgrid,
  title  = {Do Smart Grid Intrusion Detectors Learn Attacks or Operating Conditions? Evidence from a PMU-Relay Testbed and an IEEE 14-Bus Simulation},
  author = {Haider Md. Faiyaz Hussain Shahi},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/shahihussainofficial1/smartgrid-ids-generalization}
}
```

## License

Code: MIT License (see `LICENSE`). The dataset keeps its original terms of use.
