"""
Phase 3: 27-cell factorial simulation, A_threshold analysis, and sensitivity.

Uses calibrated model:
  Reliance = 0.4708 + 0.201·A − 0.20·E + 0.20·C + 0.10·(A·E) + 0.10·(A·C) + ε₁
  Error = Reliance × (1 − A) + ε₂

Outputs:
  - Console: 27-cell factorial, A_threshold table, sensitivity analysis
  - ../tables/table2_27cell_factorial.csv
  - ../tables/table3_A_threshold.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

B0, BA = 0.4708, 0.2010
BE, BC, BAE, BAC = -0.20, 0.20, 0.10, 0.10

TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)


def simulate(A, E, C, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    eps_rel = rng.normal(0, 0.05, n)
    eps_err = rng.normal(0, 0.02, n)
    r = B0 + BA*A + BE*E + BC*C + BAE*A*E + BAC*A*C + eps_rel
    r = np.clip(r, 0, 1)
    err = r*(1-A) + eps_err
    return r.mean(), r.std(), np.clip(err, 0, 1).mean(), np.clip(err, 0, 1).std()


# --- Table 2: 27-cell factorial ---
rows = []
for A in [0.5, 0.7, 0.9]:
    for E in [0.0, 0.2, 1.0]:
        for C in [0.0, 0.5, 1.0]:
            rm, rs, em, es = simulate(A, E, C)
            rows.append({
                'A': A,
                'E_years': int(E*10),
                'C': C,
                'C_label': {0.0: 'Low', 0.5: 'Moderate', 1.0: 'High'}[C],
                'Reliance_mean': round(rm, 3),
                'Reliance_sd': round(rs, 3),
                'Error_mean': round(em, 3),
                'Error_sd': round(es, 3),
            })
df27 = pd.DataFrame(rows)
print("=== 27-cell factorial ===")
print(df27.to_string(index=False))
df27.to_csv(os.path.join(TABLES_DIR, 'table2_27cell_factorial.csv'),
            index=False)


# --- Table 3: A_threshold analytical ---
def reliance_mean(A, E, C):
    return np.clip(B0 + BA*A + BE*E + BC*C + BAE*A*E + BAC*A*C, 0, 1)


def find_A_threshold(E, C, target):
    A_grid = np.linspace(0.001, 0.999, 1000)
    errs = np.array([reliance_mean(a, E, C)*(1-a) for a in A_grid])
    below = np.where(errs < target)[0]
    return float(A_grid[below[0]]) if len(below) else None


print("\n=== A_threshold table ===")
print(f"{'E_years':>8s} {'C':>10s} {'<5%':>7s} {'<10%':>7s} {'<15%':>7s} {'<20%':>7s}")
threshold_rows = []
for E in [0.0, 0.2, 1.0]:
    for C in [0.0, 0.5, 1.0]:
        c_lbl = {0.0: 'Low', 0.5: 'Moderate', 1.0: 'High'}[C]
        vals = [find_A_threshold(E, C, t) for t in [0.05, 0.10, 0.15, 0.20]]
        print(f"{int(E*10):>8d} {c_lbl:>10s} " +
              " ".join(f"{v:>7.3f}" if v else "  >1.00" for v in vals))
        threshold_rows.append({
            'E_years': int(E*10),
            'C_label': c_lbl,
            'A_threshold_5pct':  vals[0] if vals[0] is not None else np.nan,
            'A_threshold_10pct': vals[1] if vals[1] is not None else np.nan,
            'A_threshold_15pct': vals[2] if vals[2] is not None else np.nan,
            'A_threshold_20pct': vals[3] if vals[3] is not None else np.nan,
        })
pd.DataFrame(threshold_rows).to_csv(
    os.path.join(TABLES_DIR, 'table3_A_threshold.csv'), index=False)


# --- Sensitivity ---
def sim_with(A, E, C, b0, bA, bE, bC, bAE, bAC, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    e1 = rng.normal(0, 0.05, n)
    e2 = rng.normal(0, 0.02, n)
    r = b0 + bA*A + bE*E + bC*C + bAE*A*E + bAC*A*C + e1
    r = np.clip(r, 0, 1)
    return np.clip(r*(1-A)+e2, 0, 1).mean()


baseline = df27['Error_mean'].values
print("\n=== Sensitivity (±20% perturbation of fixed coefs) ===")
for name, (val, kw) in {'β_E': (BE, 'bE'), 'β_C': (BC, 'bC'),
                        'β_AE': (BAE, 'bAE'), 'β_AC': (BAC, 'bAC')}.items():
    for direction, mult in [('+20%', 1.2), ('-20%', 0.8)]:
        kwargs = {'b0': B0, 'bA': BA, 'bE': BE, 'bC': BC,
                  'bAE': BAE, 'bAC': BAC}
        kwargs[kw] = val * mult
        perturbed = [sim_with(A, E, C, **kwargs)
                     for A in [0.5, 0.7, 0.9]
                     for E in [0.0, 0.2, 1.0]
                     for C in [0.0, 0.5, 1.0]]
        rho, _ = spearmanr(baseline, perturbed)
        max_diff = max(abs(np.array(perturbed) - baseline))
        print(f"  {name} {direction}: Spearman ρ={rho:.4f}, "
              f"max |Δ|={max_diff:.3f}")

print(f"\nTables saved to {os.path.abspath(TABLES_DIR)}")
