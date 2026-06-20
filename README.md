# nursing-ai-error-thresholds

Reproducibility package for the simulation study:

> **Theoretical Exploration of Error Thresholds for Clinical AI Decision Support in Nursing: An Exploratory Simulation Study Grounded in Human–AI Reliance Data**
>
> Hiroyuki Tajima · Faculty of Nursing, Shumei University · ORCID: [0000-0003-3817-4455](https://orcid.org/0000-0003-3817-4455)

This repository contains all code, intermediate data, and figure-generation
scripts required to reproduce every numerical result and figure reported in
the manuscript, including the bootstrap confidence intervals and the
A_threshold analysis.

---

## What this study does

The paper develops an empirically calibrated simulation framework to estimate
the minimum AI accuracy (`A_threshold`) required to keep clinical error rates
below a chosen target, as a joint function of clinician experience and task
complexity. The reliance model is a linear specification:

```
Reliance = β₀ + β_A·A + β_E·E + β_C·C + β_AE·(A·E) + β_AC·(A·C) + ε₁
Error    = Reliance × (1 − A) + ε₂
```

The `A`-related coefficients (`β₀`, `β_A`) are calibrated by weighted least
squares against 9 empirical data points drawn from three independent
randomized experiments on AI-assisted decision-making (Lu and Yin, 2021;
Yin et al., 2019; total *N* = 3,502). The remaining coefficients are fixed
from the prior literature. Calibration uncertainty is quantified by
study-level bootstrap (2,000 iterations).

Key calibrated values reported in the manuscript:

| Quantity | Value |
| --- | --- |
| β₀ (intercept) | 0.4708 |
| β_A (slope on AI accuracy) | 0.201 |
| Bootstrap 95% CI for β_A | [0.023, 0.234] |
| Sign stability P(β_A > 0) | 1.000 |
| RMSE on 9 calibration points | 0.054 |

---

## Repository layout

```
nursing-ai-error-thresholds/
├── code/
│   ├── 01_luyin_analysis.py            Lu & Yin 2021 raw-data reanalysis
│   ├── 02_model_calibration.py         Weighted-least-squares calibration
│   ├── 03_simulation_and_threshold.py  27-cell factorial + A_threshold + sensitivity
│   └── 04_generate_figures.py          Bootstrap + all manuscript figures
├── figures/
│   ├── bootstrap_results.npz           2,000 bootstrap iterations of (β₀, β_A)
│   ├── figure1_conceptual.pdf          Figure 1 (vector)
│   ├── figure2_predicted_and_threshold.pdf  Figure 2, Panels A and B (vector)
│   └── figure_S1_calibration.pdf       Supplementary Figure S1 (vector)
├── tables/
│   ├── table2_27cell_factorial.csv     Predicted reliance and error per cell
│   └── table3_A_threshold.csv          Minimum AI accuracy by user × task profile
├── data/
│   └── README.md                       How to obtain the Lu & Yin raw dataset
├── LICENSE                             MIT License
├── requirements.txt                    Python dependencies
└── README.md                           This file
```

---

## How to reproduce all results

### 1. Set up a Python environment

Python ≥ 3.10 is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Re-run the Lu & Yin raw-data reanalysis

`01_luyin_analysis.py` reproduces the worker-level reliance means used as
two of the nine calibration points. The script reads
`expTwoFinalPredictionsValid1125.csv`, which is **not redistributed in this
repository** because it is the property of the original authors. To run
this script, download the file from the Lu & Yin GitHub repository (see
`data/README.md`) and place it next to the script. This step is optional;
the calibration points it produces are already hard-coded in
`02_model_calibration.py`.

```bash
cd code
python 01_luyin_analysis.py
```

### 3. Run the calibration

```bash
cd code
python 02_model_calibration.py
```

This prints `β₀ = 0.4708`, `β_A = 0.2010`, and the 9 fitted residuals.

### 4. Run the 27-cell simulation, A_threshold analysis, and sensitivity check

```bash
cd code
python 03_simulation_and_threshold.py
```

Writes `tables/table2_27cell_factorial.csv` and
`tables/table3_A_threshold.csv`, and prints the sensitivity analysis
(Spearman ρ between baseline and ±20%-perturbed cell error vectors).

### 5. Run the bootstrap and regenerate every figure

```bash
cd code
python 04_generate_figures.py
```

Writes `figures/bootstrap_results.npz` and the three PDF figures. This is
the slowest step (a few minutes) because it performs 2,000 bootstrap
iterations.

All scripts use a fixed random seed (`42`), so re-running them on a
different machine produces bit-identical numerical output up to platform-
specific floating-point ordering.

---

## LLM accuracy operating range

To contextualize the simulation's operating range, the manuscript characterizes
the accuracy of contemporary general-purpose large language models on complex
clinical tasks using published, peer-reviewed benchmarks rather than an in-house
evaluation. The reference range of approximately 0.5–0.7 used in the figures is
drawn from Hager et al. (2024, *Nature Medicine*) and Eriksen et al. (2024,
*NEJM AI*); for example, GPT-4 achieved 57% correct diagnoses on complex
clinicopathological cases. No proprietary case set is distributed with this
repository; the only inputs to the calibration are the published human–AI
reliance data points listed below.


---

## Software environment used for the published results

| Package | Version |
| --- | --- |
| Python | 3.11 |
| NumPy | 1.26 |
| SciPy | 1.13 |
| pandas | 2.2 |
| matplotlib | 3.8 |

Newer versions are likely to work without modification; `requirements.txt`
specifies minimum versions rather than exact pins.

---

## Citation

If you use this code or build on the framework, please cite the manuscript:

```
Tajima, H. (2026). Theoretical Exploration of Error Thresholds for Clinical
AI Decision Support in Nursing: An Exploratory Simulation Study Grounded in
Human–AI Reliance Data. [Journal name and DOI to be added upon publication.]
```

A BibTeX entry will be added here once the manuscript is accepted.

---

## Calibration-data sources

The 9 empirical reliance points used for calibration come from:

- **Lu, Z., & Yin, M. (2021).** Human reliance on machine learning models when
  performance feedback is limited: Heuristics and risks. *CHI '21*.
  Raw data: <https://github.com/ZhuoranLu/Trustworthy-ML>
- **Yin, M., Wortman Vaughan, J., & Wallach, H. (2019).** Understanding the
  effect of accuracy on trust in machine learning models. *CHI '19*.
  Reliance values were extracted from Figures 3a and 5a of the published paper.

See the manuscript Methods section for the precise extraction procedure and
the rationale for treating these data as appropriate calibration targets.

---

## License

This repository is released under the [MIT License](LICENSE). The
calibration data points hard-coded in `02_model_calibration.py` are derived
from the cited works above and remain subject to the original authors'
licensing terms.

---

## Contact

Questions, bug reports, and requests for the 100 clinical case scenarios
should be addressed to:

**Hiroyuki Tajima**
Faculty of Nursing, Shumei University
1-1 Daigaku-cho, Yachiyo, Chiba 276-0003, Japan
Email: tajima@mailg.shumei-u.ac.jp
ORCID: <https://orcid.org/0000-0003-3817-4455>
