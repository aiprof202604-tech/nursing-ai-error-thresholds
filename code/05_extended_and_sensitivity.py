"""
Round-2 revision analyses for JMIR Nursing #99590.

Addresses editor comments (Decision B, minor revision):
  (2) A more realistic JOINT human-AI decision model that accounts for the
      clinician's own accuracy, independent judgment, correction of a wrong
      AI recommendation, and errors made when AI advice is rejected.
  (3) MUCH WIDER sensitivity analysis: wide one-at-a-time ranges, sign
      reversals (alternative coefficient directions), a global Monte-Carlo
      sweep over all four literature-fixed coefficients, and alternative
      model STRUCTURES (joint model + logistic reliance).

The primary (locked) results are NOT changed. Everything here is additive.
Base calibrated model recovered as a limiting case is verified first.

Design notes (per project coding conventions):
  * Resume: each stage writes a CSV; re-running skips completed stages unless
    --force is passed.
  * Progress:每 stage prints a header and a lightweight progress line.
  * Speed: all A_threshold evaluations are closed-form grid searches and are
    fully vectorised over Monte-Carlo draws (no Python inner loops).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locked calibrated coefficients (immutable; see 02/03 scripts)
# ---------------------------------------------------------------------------
B0, BA = 0.4708, 0.2010
BE, BC, BAE, BAC = -0.20, 0.20, 0.10, 0.10

CELLS = [(E, C) for E in (0.0, 0.2, 1.0) for C in (0.0, 0.5, 1.0)]
CLBL = {0.0: "Low", 0.5: "Moderate", 1.0: "High"}
TARGETS = [0.05, 0.10, 0.15, 0.20]

OUT = os.path.join(os.path.dirname(__file__), "..", "tables")
os.makedirs(OUT, exist_ok=True)
A_GRID = np.linspace(0.001, 0.999, 1000)


def reliance_mean(A, E, C, b0=B0, bA=BA, bE=BE, bC=BC, bAE=BAE, bAC=BAC):
    """Calibrated linear reliance, clipped to [0,1]. Broadcasts over arrays."""
    r = b0 + bA * A + bE * E + bC * C + bAE * (A * E) + bAC * (A * C)
    return np.clip(r, 0.0, 1.0)


# ---------------------------------------------------------------------------
# BASE model A_threshold (sanity check vs locked values)
# ---------------------------------------------------------------------------
def base_error(A, E, C):
    return reliance_mean(A, E, C) * (1 - A)


def a_threshold(err_of_A, target):
    """Smallest A in grid with mean error < target; None if never."""
    errs = err_of_A(A_GRID)
    below = np.where(errs < target)[0]
    return float(A_GRID[below[0]]) if below.size else None


def stage_base_check():
    print("\n[Stage 0] Base-model A_threshold sanity check (must match locked)")
    rows = []
    for E, C in CELLS:
        vals = [a_threshold(lambda a, E=E, C=C: base_error(a, E, C), t)
                for t in TARGETS]
        rows.append((int(E * 10), CLBL[C], *vals))
        print(f"   E={int(E*10):>2d}y {CLBL[C]:>8s}: " +
              " ".join(f"{v:.3f}" if v else ">1.00" for v in vals))
    key = [r for r in rows if r[0] == 0 and r[1] == "High"][0]
    assert abs(key[3] - 0.894) < 0.002 and abs(key[5] - 0.779) < 0.002, \
        "Base A_threshold drift — STOP."
    print("   OK: novice x high = 0.894 (<10%), 0.779 (<20%) reproduced.")


# ---------------------------------------------------------------------------
# (2) JOINT human-AI decision model
#     Error_joint = R*(1-A) + (1-R)*(1-p_c)
#     p_c = clinician independent accuracy. p_c -> 1 recovers the base model.
# ---------------------------------------------------------------------------
def joint_error(A, E, C, p_c):
    R = reliance_mean(A, E, C)
    return np.clip(R * (1 - A) + (1 - R) * (1 - p_c), 0.0, 1.0)


def joint_error_floor(E, C, p_c):
    """Irreducible team error as A->1: (1-R(1,E,C))*(1-p_c)."""
    R1 = reliance_mean(1.0, E, C)
    return (1 - R1) * (1 - p_c)


def stage_joint(force=False):
    path = os.path.join(OUT, "r2_joint_model_threshold.csv")
    if os.path.exists(path) and not force:
        print("\n[Stage 1] joint model: cached -> skip")
        return pd.read_csv(path)
    print("\n[Stage 1] Joint-decision model across clinician accuracy p_c")
    pcs = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]
    rows = []
    for i, p_c in enumerate(pcs, 1):
        for E, C in CELLS:
            floor = joint_error_floor(E, C, p_c)
            vals = []
            for t in TARGETS:
                v = a_threshold(lambda a, E=E, C=C, p=p_c: joint_error(a, E, C, p), t)
                vals.append(v if v is not None else np.nan)
            rows.append({"p_c": p_c, "E_years": int(E * 10), "C": CLBL[C],
                         "error_floor": round(floor, 4),
                         "A_thr_5pct": vals[0], "A_thr_10pct": vals[1],
                         "A_thr_15pct": vals[2], "A_thr_20pct": vals[3]})
        print(f"   p_c={p_c:.2f} done ({i}/{len(pcs)})")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    # Key-cell summary for the manuscript
    print("\n   Novice x High-complexity (highest-risk cell):")
    sub = df[(df.E_years == 0) & (df.C == "High")]
    for _, r in sub.iterrows():
        f = "%.1f%%" % (r.error_floor * 100)
        t10 = "n/a" if pd.isna(r.A_thr_10pct) else f"{r.A_thr_10pct:.3f}"
        t20 = "n/a" if pd.isna(r.A_thr_20pct) else f"{r.A_thr_20pct:.3f}"
        print(f"     p_c={r.p_c:.2f}: floor={f:>6s}  A_thr(<10%)={t10}  A_thr(<20%)={t20}")
    print("\n   Experienced x Low-complexity (lowest-deference cell):")
    sub = df[(df.E_years == 10) & (df.C == "Low")]
    for _, r in sub.iterrows():
        f = "%.1f%%" % (r.error_floor * 100)
        t10 = "n/a" if pd.isna(r.A_thr_10pct) else f"{r.A_thr_10pct:.3f}"
        t20 = "n/a" if pd.isna(r.A_thr_20pct) else f"{r.A_thr_20pct:.3f}"
        print(f"     p_c={r.p_c:.2f}: floor={f:>6s}  A_thr(<10%)={t10}  A_thr(<20%)={t20}")
    return df


# ---------------------------------------------------------------------------
# (2b) ASYMMETRIC-reliance variant: deference to an INCORRECT AI is scaled by
#      rho (<=1), capturing the clinician's ability to catch/correct a wrong AI
#      (empirically: reliance stronger for correct than incorrect advice).
#      Error = (1-A)*[rho*R + (1-rho*R)*(1-p_c)] + A*(1-R)*(1-p_c)
#      rho=1 reduces to the deference joint model above.
# ---------------------------------------------------------------------------
def joint_error_asym(A, E, C, p_c, rho):
    R = reliance_mean(A, E, C)
    Rw = np.clip(rho * R, 0.0, 1.0)              # deference to a WRONG AI
    err_ai_wrong = Rw * 1.0 + (1 - Rw) * (1 - p_c)
    err_ai_right = (1 - R) * (1 - p_c)           # reject a correct AI -> own
    return np.clip((1 - A) * err_ai_wrong + A * err_ai_right, 0.0, 1.0)


def stage_asym(force=False):
    path = os.path.join(OUT, "r2_asymmetric_threshold.csv")
    if os.path.exists(path) and not force:
        print("\n[Stage 2] asymmetric model: cached -> skip")
        return pd.read_csv(path)
    print("\n[Stage 2] Asymmetric-reliance variant (correction of wrong AI)")
    rows = []
    for p_c in (1.00, 0.90, 0.80):
        for rho in (1.0, 0.7, 0.5):
            for E, C in CELLS:
                vals = [a_threshold(
                    lambda a, E=E, C=C, p=p_c, rr=rho: joint_error_asym(a, E, C, p, rr), t)
                    for t in TARGETS]
                rows.append({"p_c": p_c, "rho": rho, "E_years": int(E * 10),
                             "C": CLBL[C],
                             "A_thr_10pct": vals[1] if vals[1] else np.nan,
                             "A_thr_20pct": vals[3] if vals[3] else np.nan})
        print(f"   p_c={p_c:.2f} done")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    sub = df[(df.E_years == 0) & (df.C == "High")]
    print("   Novice x High, A_thr(<20%) by (p_c, rho):")
    for _, r in sub.iterrows():
        t = "n/a" if pd.isna(r.A_thr_20pct) else f"{r.A_thr_20pct:.3f}"
        print(f"     p_c={r.p_c:.2f} rho={r.rho:.1f}: {t}")
    return df


# ---------------------------------------------------------------------------
# (3a) WIDE one-at-a-time sensitivity: -100%, -50%, +50%, +100%, sign flip, 0
# ---------------------------------------------------------------------------
def stage_wide_oat(force=False):
    from scipy.stats import spearmanr
    path = os.path.join(OUT, "r2_wide_oat_sensitivity.csv")
    if os.path.exists(path) and not force:
        print("\n[Stage 3] wide OAT: cached -> skip")
        return pd.read_csv(path)
    print("\n[Stage 3] Wide one-at-a-time sensitivity (incl. sign reversal)")
    base_grid = np.array([base_error(np.array([a]), E, C)[0]
                          for a in [0.5, 0.7, 0.9]
                          for E, C in CELLS]).reshape(-1)  # 27 baseline errors

    def all27_errors(bE, bC, bAE, bAC):
        out = []
        for A in (0.5, 0.7, 0.9):
            for E, C in CELLS:
                r = reliance_mean(A, E, C, bE=bE, bC=bC, bAE=bAE, bAC=bAC)
                out.append(float(np.clip(r * (1 - A), 0, 1)))
        return np.array(out)

    def key_thr(bE, bC, bAE, bAC, target):
        return a_threshold(
            lambda a: reliance_mean(a, 0.0, 1.0, bE=bE, bC=bC, bAE=bAE, bAC=bAC) * (1 - a),
            target)

    coefs = {"beta_E": (BE, 0), "beta_C": (BC, 1),
             "beta_AE": (BAE, 2), "beta_AC": (BAC, 3)}
    factors = {"-100%": -1.0, "-50%": 0.5, "0": 0.0,
               "+50%": 1.5, "+100%": 2.0, "sign_flip": -1.0}
    rows = []
    base_vec = [BE, BC, BAE, BAC]
    for cname, (val, idx) in coefs.items():
        for fname, f in factors.items():
            vec = list(base_vec)
            vec[idx] = -val if fname == "sign_flip" else val * f
            errs = all27_errors(*vec)
            rho, _ = spearmanr(base_grid, errs)
            mx = float(np.max(np.abs(errs - base_grid)))
            t10 = key_thr(*vec, 0.10)
            t20 = key_thr(*vec, 0.20)
            rows.append({"coef": cname, "perturbation": fname,
                         "value": round(vec[idx], 3), "spearman_rho": round(rho, 4),
                         "max_abs_delta": round(mx, 3),
                         "novhigh_A_thr_10": None if t10 is None else round(t10, 3),
                         "novhigh_A_thr_20": None if t20 is None else round(t20, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# (3b) GLOBAL Monte-Carlo sensitivity: sample ALL four fixed coefficients
#      simultaneously from WIDE uniform priors that INCLUDE SIGN REVERSALS.
#      Report distribution of key A_threshold values. Fully vectorised.
# ---------------------------------------------------------------------------
def stage_global(force=False, n=50000, seed=42):
    path = os.path.join(OUT, "r2_global_sensitivity_summary.csv")
    if os.path.exists(path) and not force:
        print("\n[Stage 4] global MC: cached -> skip")
        return pd.read_csv(path)
    print(f"\n[Stage 4] Global MC sensitivity (n={n}/prior)")
    rng = np.random.default_rng(seed)
    Agr = A_GRID[None, :]

    def draw(prior):
        if prior == "signed":     # wide magnitude, theoretical signs preserved
            return (rng.uniform(-0.40, 0.0, n), rng.uniform(0.0, 0.40, n),
                    rng.uniform(0.0, 0.20, n), rng.uniform(0.0, 0.20, n))
        else:                     # fully agnostic: sign reversals allowed
            return (rng.uniform(-0.40, 0.40, n), rng.uniform(-0.40, 0.40, n),
                    rng.uniform(-0.20, 0.20, n), rng.uniform(-0.20, 0.20, n))

    def key_thr_vec(E, C, target, bE, bC, bAE, bAC):
        r = (B0 + BA * Agr + bE[:, None] * E + bC[:, None] * C
             + bAE[:, None] * (Agr * E) + bAC[:, None] * (Agr * C))
        r = np.clip(r, 0, 1)
        err = r * (1 - Agr)
        mask = err < target
        idx = np.argmax(mask, axis=1)
        thr = A_GRID[idx].astype(float)
        thr[~mask.any(axis=1)] = np.nan
        return thr

    summary = []
    scen = [(0.0, 1.0, "novice_high"), (1.0, 0.0, "experienced_low")]
    for prior in ("signed", "agnostic"):
        bE, bC, bAE, bAC = draw(prior)
        print(f"   -- prior='{prior}'")
        for E, C, label in scen:
            for t in (0.10, 0.20):
                thr = key_thr_vec(E, C, t, bE, bC, bAE, bAC)
                valid = thr[~np.isnan(thr)]
                # qualitative anchor: still needs 'substantial' accuracy A>=0.70
                anchor = float(np.mean(valid >= 0.70)) if valid.size else 0.0
                row = {"prior": prior, "scenario": label, "target": t,
                       "median": round(float(np.median(valid)), 3),
                       "p05": round(float(np.percentile(valid, 5)), 3),
                       "p95": round(float(np.percentile(valid, 95)), 3),
                       "frac_reachable": round(float(valid.size / thr.size), 3),
                       "frac_need_A_ge_0.70": round(anchor, 3)}
                summary.append(row)
                print(f"      {label:16s} <{int(t*100)}%: "
                      f"median={row['median']:.3f} [{row['p05']:.3f},{row['p95']:.3f}] "
                      f"P(A_thr>=.70)={row['frac_need_A_ge_0.70']:.3f}")
    df = pd.DataFrame(summary)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# (3c) ALTERNATIVE STRUCTURE: logistic reliance form
#      R = clip(1/(1+exp(-k*(A - A0))) shifted/scaled to match calibration range
# ---------------------------------------------------------------------------
def stage_logistic(force=False):
    from scipy.optimize import differential_evolution
    path = os.path.join(OUT, "r2_altstructure_threshold.csv")
    if os.path.exists(path) and not force:
        print("\n[Stage 5] alt structures: cached -> skip")
        return pd.read_csv(path)
    print("\n[Stage 5] Alternative reliance STRUCTURES (bounded fits)")
    # calibration targets (E=0, C=0.5) from 02_model_calibration.py
    A_obs = np.array([0.50, 0.80, 0.55, 0.80, 1.00, 0.60, 0.70, 0.90, 0.95])
    R_obs = np.array([0.642, 0.664, 0.795, 0.830, 0.833, 0.745, 0.760, 0.800, 0.820])
    nvec = np.array([252, 214, 170, 170, 170, 200, 200, 200, 200], float)
    w = np.sqrt(nvec)

    def sigmoid(A, Rmin, Rmax, k, A0):
        return Rmin + (Rmax - Rmin) / (1 + np.exp(-k * (A - A0)))

    def sat(A, base, c, k):                      # concave saturating (exp)
        return base + c * (1 - np.exp(-k * A))

    def fit(func, bounds):
        def loss(p):
            return np.sum((w * (func(A_obs, *p) - R_obs)) ** 2)
        r = differential_evolution(loss, bounds, seed=42, tol=1e-10,
                                   maxiter=2000, polish=True)
        pred = func(A_obs, *r.x)
        rmse = float(np.sqrt(np.mean((pred - R_obs) ** 2)))
        return r.x, rmse

    p_sig, rmse_sig = fit(sigmoid, [(0.3, 0.6), (0.7, 0.98), (2, 20), (0.2, 0.9)])
    p_sat, rmse_sat = fit(sat, [(0.0, 0.7), (0.05, 0.6), (0.5, 20)])
    print(f"   sigmoid  RMSE={rmse_sig:.4f}  params={np.round(p_sig,3)}")
    print(f"   saturating RMSE={rmse_sat:.4f}  params={np.round(p_sat,3)}")
    print(f"   (linear form RMSE=0.054)")

    def thr_for(func, params):
        def rel(A, E, C):
            return np.clip(func(A, *params) + BE * E + BC * C, 0, 1)
        out = {}
        for E, C in CELLS:
            vals = [a_threshold(lambda a, E=E, C=C: rel(a, E, C) * (1 - a), t)
                    for t in TARGETS]
            out[(int(E * 10), CLBL[C])] = vals
        return out

    rows = []
    for name, func, params, rmse in [("sigmoid", sigmoid, p_sig, rmse_sig),
                                     ("saturating", sat, p_sat, rmse_sat)]:
        thr = thr_for(func, params)
        for (E, C), vals in thr.items():
            rows.append({"structure": name, "rmse": round(rmse, 4),
                         "E_years": E, "C": C,
                         "A_thr_10": vals[1], "A_thr_20": vals[3]})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    for name in ("sigmoid", "saturating"):
        k = df[(df.structure == name) & (df.E_years == 0) & (df.C == "High")].iloc[0]
        t10 = "n/a" if pd.isna(k.A_thr_10) else f"{k.A_thr_10:.3f}"
        t20 = "n/a" if pd.isna(k.A_thr_20) else f"{k.A_thr_20:.3f}"
        print(f"   Novice x High ({name}): A_thr(<10%)={t10}, A_thr(<20%)={t20} "
              f"(linear: 0.894 / 0.779)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute all stages")
    args = ap.parse_args()
    print("=" * 70)
    print("JMIR #99590 Round-2 extended analyses (joint model + wide sensitivity)")
    print("=" * 70)
    stage_base_check()
    stage_joint(args.force)
    stage_asym(args.force)
    stage_wide_oat(args.force)
    stage_global(args.force)
    stage_logistic(args.force)
    print("\nAll stages complete. CSVs in:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
