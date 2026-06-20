# Data sources

## Lu & Yin (2021) raw data

`01_luyin_analysis.py` requires the file `expTwoFinalPredictionsValid1125.csv`
from the original authors' public repository:

> <https://github.com/ZhuoranLu/Trustworthy-ML>

This file is **not redistributed here** because it is the property of the
original authors and remains subject to their licensing terms. To run the
reanalysis script:

1. Download `expTwoFinalPredictionsValid1125.csv` from the URL above.
2. Place it in the `code/` directory next to `01_luyin_analysis.py`.
3. Run `python 01_luyin_analysis.py` from the `code/` directory.

This step is **optional**. The two worker-level reliance means it produces
(M = 0.642 at A = 0.50, N = 252; M = 0.664 at A = 0.80, N = 214) are already
hard-coded as calibration targets in `02_model_calibration.py`, so the rest
of the analysis pipeline runs without this file.

## Yin et al. (2019) reliance values

The seven reliance values from Yin et al. (2019), Experiment 1 Phase 1 and
Experiment 3 Phase 2, were extracted from the published Figures 3a and 5a by
visual inspection. The procedure is documented in the manuscript Methods
section. The extracted values are hard-coded in `02_model_calibration.py`.


