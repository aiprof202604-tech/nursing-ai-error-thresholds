"""
Phase 4: Bootstrap calibration uncertainty quantification and figure generation.

Inputs (all hardcoded from prior phases):
  - 9 empirical reliance data points from Lu & Yin (2021) and Yin et al. (2019)
  - Calibrated point estimates: β₀ = 0.4708, β_A = 0.201
  - Literature-fixed coefficients: β_E = -0.20, β_C = 0.20, β_AE = 0.10, β_AC = 0.10

Outputs (relative to repository root):
  - figures/bootstrap_results.npz             (2,000 iterations of (β₀, β_A))
  - figures/figure1_conceptual.pdf            (conceptual model, vector)
  - figures/figure2_predicted_and_threshold.pdf  (Panel A + Panel B, vector)
  - figures/figure_S1_calibration.pdf         (calibration fit, vector)

Reproducibility:
  - Random seed: 42
  - Python ≥ 3.10, NumPy, SciPy ≥ 1.13, matplotlib ≥ 3.7
  - Run from the code/ directory: python 04_generate_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.optimize import minimize

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Calibration setup (mirrors 02_model_calibration.py)
# ------------------------------------------------------------------
BE, BC, BAE, BAC = -0.20, 0.20, 0.10, 0.10  # literature-fixed
B0_POINT, BA_POINT = 0.4708, 0.2010          # calibrated point estimates

# 9 empirical data points, grouped by study for study-level bootstrap
STUDIES = {
    'Lu&Yin_Exp2': [
        {'A': 0.50, 'E': 0.0, 'C': 0.5, 'obs': 0.642, 'n': 252},
        {'A': 0.80, 'E': 0.0, 'C': 0.5, 'obs': 0.664, 'n': 214},
    ],
    'Yin19_Exp3_P2': [
        {'A': 0.55, 'E': 0.0, 'C': 0.5, 'obs': 0.795, 'n': 170},
        {'A': 0.80, 'E': 0.0, 'C': 0.5, 'obs': 0.830, 'n': 170},
        {'A': 1.00, 'E': 0.0, 'C': 0.5, 'obs': 0.833, 'n': 170},
    ],
    'Yin19_Exp1_P1': [
        {'A': 0.60, 'E': 0.0, 'C': 0.5, 'obs': 0.745, 'n': 200},
        {'A': 0.70, 'E': 0.0, 'C': 0.5, 'obs': 0.760, 'n': 200},
        {'A': 0.90, 'E': 0.0, 'C': 0.5, 'obs': 0.800, 'n': 200},
        {'A': 0.95, 'E': 0.0, 'C': 0.5, 'obs': 0.820, 'n': 200},
    ],
}
ALL_TARGETS = [t for ts in STUDIES.values() for t in ts]


def reliance_pred(A, E, C, b0, bA):
    rel = b0 + bA * A + BE * E + BC * C + BAE * (A * E) + BAC * (A * C)
    return np.clip(rel, 0, 1)


def calibrate(targets, x0=(0.5, 0.3)):
    def loss(params):
        b0, bA = params
        ss = 0.0
        for t in targets:
            pred = reliance_pred(t['A'], t['E'], t['C'], b0, bA)
            w = np.sqrt(t['n'])
            ss += (w * (pred - t['obs'])) ** 2
        return ss
    res = minimize(loss, x0=list(x0), method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-8})
    return res.x


# ------------------------------------------------------------------
# 2. Verify point estimates match published values
# ------------------------------------------------------------------
b0_check, bA_check = calibrate(ALL_TARGETS)
assert abs(b0_check - B0_POINT) < 1e-3, f"β₀ mismatch: {b0_check} vs {B0_POINT}"
assert abs(bA_check - BA_POINT) < 1e-3, f"β_A mismatch: {bA_check} vs {BA_POINT}"
print(f"Point estimate verified: β₀ = {b0_check:.4f}, β_A = {bA_check:.4f}")

obs_arr = np.array([t['obs'] for t in ALL_TARGETS])
pred_arr = np.array([reliance_pred(t['A'], t['E'], t['C'], B0_POINT, BA_POINT)
                     for t in ALL_TARGETS])
rmse_point = float(np.sqrt(np.mean((obs_arr - pred_arr) ** 2)))
print(f"Point-estimate RMSE = {rmse_point:.4f}")

# ------------------------------------------------------------------
# 3. Study-level bootstrap (2,000 iterations)
# ------------------------------------------------------------------
N_BOOT = 2000
rng = np.random.default_rng(42)
study_names = list(STUDIES.keys())
boot_b0 = np.empty(N_BOOT)
boot_bA = np.empty(N_BOOT)

for i in range(N_BOOT):
    sampled = rng.choice(study_names, size=len(study_names), replace=True)
    targets = []
    for s in sampled:
        targets.extend(STUDIES[s])
    try:
        b0_i, bA_i = calibrate(targets)
    except Exception:
        b0_i, bA_i = np.nan, np.nan
    boot_b0[i] = b0_i
    boot_bA[i] = bA_i

valid = ~np.isnan(boot_b0)
boot_b0 = boot_b0[valid]
boot_bA = boot_bA[valid]
print(f"Bootstrap iterations: {N_BOOT} requested, {len(boot_b0)} valid")
print(f"β_A bootstrap 95% CI: [{np.percentile(boot_bA, 2.5):.3f}, "
      f"{np.percentile(boot_bA, 97.5):.3f}]")
print(f"β_A sign stability P(β_A > 0) = {(boot_bA > 0).mean():.3f}")

np.savez(os.path.join(FIG_DIR, 'bootstrap_results.npz'),
         boot_b0=boot_b0, boot_bA=boot_bA,
         b0_point=B0_POINT, bA_point=BA_POINT,
         BE=BE, BC=BC, BAE=BAE, BAC=BAC,
         rmse_point=rmse_point)
print("Saved: bootstrap_results.npz")

# ------------------------------------------------------------------
# 4. A_threshold helper
# ------------------------------------------------------------------
def a_threshold(target_err, E, C, b0, bA, A_grid=None):
    """Smallest A in [0, 1] for which Reliance(A,E,C)*(1-A) <= target_err."""
    if A_grid is None:
        A_grid = np.linspace(0.001, 0.999, 1000)
    rel = np.clip(b0 + bA * A_grid + BE * E + BC * C
                  + BAE * (A_grid * E) + BAC * (A_grid * C), 0, 1)
    err = rel * (1 - A_grid)
    below = np.where(err <= target_err)[0]
    return float(A_grid[below[0]]) if len(below) else np.nan


# ------------------------------------------------------------------
# 5. Global matplotlib style
# ------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'pdf.fonttype': 42,   # TrueType, editable in Illustrator
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
})

# ------------------------------------------------------------------
# 6. Figure 1 — Conceptual model
# ------------------------------------------------------------------
fig1, ax = plt.subplots(figsize=(9, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title('Figure 1. Conceptual model of AI-assisted clinical decision-making',
             fontsize=12, fontweight='bold', pad=10)

def box(x, y, w, h, text, fc, ec, fontsize=10, weight='bold'):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.06,rounding_size=0.12",
                       linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight)

# Three input boxes
box(0.2, 3.6, 1.7, 0.8, 'AI accuracy\n(A)',         '#E3F0FB', '#3A78B5')
box(0.2, 2.3, 1.7, 0.8, 'Clinician\nexperience (E)', '#E3F0FB', '#3A78B5')
box(0.2, 1.0, 1.7, 0.8, 'Task\ncomplexity (C)',     '#E3F0FB', '#3A78B5')

# Reliance box (center) with equation
box(3.3, 1.7, 3.6, 2.0, '', '#FFF4D6', '#D9A21B')
ax.text(5.1, 3.25, 'Reliance', ha='center', va='center',
        fontsize=11, fontweight='bold')
ax.text(5.1, 2.55,
        r'$\beta_0 + \beta_A A + \beta_E E + \beta_C C$' + '\n'
        + r'$+\ \beta_{AE}(A{\cdot}E) + \beta_{AC}(A{\cdot}C)$',
        ha='center', va='center', fontsize=9.5)

# Error box
box(7.5, 2.3, 1.9, 1.0, 'Error\n' + r'Reliance × (1 − A)',
    '#FBE3E3', '#C24A4A')

# Clinical acceptability
box(7.5, 0.6, 1.9, 1.0, 'Clinical\nacceptability\n(Error < target)',
    '#E3F5E3', '#3FAE5F', fontsize=9)

# Arrows: inputs → reliance
arrow_kw = dict(arrowstyle='->', mutation_scale=14, lw=1.4, color='#3A78B5')
for y in (4.0, 2.7, 1.4):
    ax.add_patch(FancyArrowPatch((1.95, y), (3.25, 2.7), **arrow_kw))

# reliance → error
ax.add_patch(FancyArrowPatch((6.95, 2.7), (7.45, 2.8),
                             arrowstyle='->', mutation_scale=14,
                             lw=1.6, color='#D9A21B'))

# (1-A) factor: A box → error (curved, red)
ax.add_patch(FancyArrowPatch((1.05, 4.4), (7.5, 3.0),
                             arrowstyle='->', mutation_scale=12,
                             lw=1.2, color='#C24A4A',
                             connectionstyle="arc3,rad=-0.18"))
ax.text(4.5, 4.55, '(1 − A) factor', ha='center', va='center',
        fontsize=9, style='italic', color='#C24A4A')

# error → clinical acceptability
ax.add_patch(FancyArrowPatch((8.45, 2.3), (8.45, 1.6),
                             arrowstyle='->', mutation_scale=14,
                             lw=1.4, color='#3FAE5F'))

# Calibration footnote
ax.text(5.1, 0.85,
        r'$\beta_0,\ \beta_A$ calibrated from 9 empirical points'
        '\n(Lu & Yin 2021; Yin et al. 2019; ' + r'$N \approx 1{,}776$)',
        ha='center', va='center', fontsize=8.5, style='italic',
        color='#555555',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#BBBBBB', linestyle='--', linewidth=0.8))

fig1.savefig(os.path.join(FIG_DIR, 'figure1_conceptual.pdf'))
plt.close(fig1)
print("Saved: figure1_conceptual.pdf")

# ------------------------------------------------------------------
# 7. Figure 2 — Panel A + Panel B (combined)
# ------------------------------------------------------------------
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0))

# ---- Panel A: predicted error curves for high-complexity tasks ----
A_grid = np.linspace(0.0, 1.0, 401)
exp_levels = [(0.0, 'Novice (0 years)',     '#C0392B', '-',  1.8),
              (0.2, 'Junior (2 years)',     '#E67E22', '--', 1.6),
              (1.0, 'Experienced (10 yrs)', '#27AE60', '-.', 1.6)]
C_high = 1.0

for E, lbl, color, ls, lw in exp_levels:
    rel = np.clip(B0_POINT + BA_POINT * A_grid + BE * E + BC * C_high
                  + BAE * (A_grid * E) + BAC * (A_grid * C_high), 0, 1)
    err = rel * (1 - A_grid)
    axA.plot(A_grid, err, color=color, ls=ls, lw=lw, label=lbl)

# literature reference range for current general-purpose LLM accuracy on
# complex clinical tasks (~0.5-0.7; Hager et al. 2024 Nat Med; Eriksen et al. 2024 NEJM AI)
axA.axvspan(0.5, 0.7, color='#5B8DB8', alpha=0.15, zorder=0)

# error reference lines
for y, label, color in [(0.30, '30% error', '#C0392B'),
                        (0.20, '20% error', '#E67E22'),
                        (0.10, '10% error', '#27AE60')]:
    axA.axhline(y, color=color, ls=':', lw=0.9, alpha=0.7)
    axA.text(0.012, y + 0.008, label, fontsize=8, color=color, alpha=0.95)

# Reference-range annotation: placed below-right with a leader line into the band
axA.annotate('Current general-purpose\nLLM range (complex tasks)',
             xy=(0.60, 0.17), xytext=(0.83, 0.055),
             fontsize=8, color='#2C5775', ha='center', va='center',
             style='italic',
             arrowprops=dict(arrowstyle='-', color='#5B8DB8', lw=0.8))

axA.set_xlim(0, 1)
axA.set_ylim(0, 0.55)
axA.set_xlabel('AI accuracy (A)')
axA.set_ylabel('Predicted error rate')
axA.set_title('(A) High-complexity clinical tasks', fontweight='bold')
axA.legend(loc='upper right', framealpha=0.95)
axA.grid(True, alpha=0.3)

# ---- Panel B: A_threshold curves with bootstrap CI bands ----
err_grid = np.linspace(0.05, 0.30, 51)
profiles = [
    ('Novice × High',       0.0, 1.0, '#C0392B', '-',  1.9),
    ('Novice × Moderate',   0.0, 0.5, '#E67E22', '-',  1.6),
    ('Junior × High',       0.2, 1.0, '#7E57C2', '--', 1.6),
    ('Experienced × High',  1.0, 1.0, '#27AE60', '-.', 1.6),
    ('Experienced × Low',   1.0, 0.0, '#2980B9', ':',  2.0),
]

for lbl, E, C, color, ls, lw in profiles:
    # Point estimate
    a_pt = np.array([a_threshold(t, E, C, B0_POINT, BA_POINT) for t in err_grid])

    # Bootstrap CI
    boot_curves = np.empty((len(boot_bA), len(err_grid)))
    for i, (b0_i, bA_i) in enumerate(zip(boot_b0, boot_bA)):
        boot_curves[i] = [a_threshold(t, E, C, b0_i, bA_i) for t in err_grid]
    lo = np.nanpercentile(boot_curves, 2.5, axis=0)
    hi = np.nanpercentile(boot_curves, 97.5, axis=0)

    axB.fill_between(err_grid * 100, lo, hi, color=color, alpha=0.18, linewidth=0)
    axB.plot(err_grid * 100, a_pt, color=color, ls=ls, lw=lw, label=lbl)

# reference lines for frontier and observed accuracy
for y, label in [(0.95, 'Demanding (0.95)'),
                 (0.85, 'Near-term (0.85)'),
                 (0.70, 'Reference (0.70)')]:
    axB.axhline(y, color='#777777', ls=':', lw=0.8, alpha=0.7)
    axB.text(29.7, y + 0.006, label, fontsize=8, color='#555555',
             ha='right', va='bottom', style='italic')

axB.set_xlim(5, 30)
axB.set_ylim(0.40, 1.00)
axB.set_xlabel('Clinically acceptable error threshold (%)')
axB.set_ylabel(r'Required AI accuracy ($A_{threshold}$)')
axB.set_title('(B) Minimum AI accuracy with bootstrap 95% CI bands',
              fontweight='bold')
axB.legend(loc='lower left', framealpha=0.95)
axB.grid(True, alpha=0.3)

# Figure number/caption intentionally NOT embedded in the image (per JMIR figure rules)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'figure2_predicted_and_threshold.pdf'))
plt.close(fig2)
print("Saved: figure2_predicted_and_threshold.pdf")

# ------------------------------------------------------------------
# 8. Figure S1 — Calibration fit
# ------------------------------------------------------------------
figS, axS = plt.subplots(figsize=(8, 5.5))

study_styles = {
    'Lu&Yin_Exp2':   ('#C0392B', 'o', 'Lu & Yin 2021, Exp 2',           466),
    'Yin19_Exp3_P2': ('#27AE60', 's', 'Yin et al. 2019, Exp 3, Phase 2', 1042),
    'Yin19_Exp1_P1': ('#7E57C2', '^', 'Yin et al. 2019, Exp 1, Phase 1', 1994),
}

# Calibrated reliance line at E=0, C=0.5
A_line = np.linspace(0.45, 1.0, 200)
rel_line = B0_POINT + BA_POINT * A_line + BC * 0.5
axS.plot(A_line, rel_line, color='#2C7FB8', lw=2.0,
         label=f'Calibrated model (E=0, C=0.5)')

for sname, points in STUDIES.items():
    color, marker, lbl, n_total = study_styles[sname]
    xs = [p['A'] for p in points]
    ys = [p['obs'] for p in points]
    axS.scatter(xs, ys, s=110, c=color, marker=marker,
                edgecolors='black', linewidths=0.9,
                label=f'{lbl} (N={n_total:,})', zorder=3)

axS.set_xlim(0.45, 1.02)
axS.set_ylim(0.55, 0.90)
axS.set_xlabel('AI accuracy (A)')
axS.set_ylabel('Reliance (agreement / final-agreement fraction)')
axS.set_title('Calibration of the linear reliance model against '
              '9 empirical points\nfrom 3 independent studies '
              r'($N \approx 1{,}776$;'
              + f' RMSE = {rmse_point:.3f})',
              fontweight='bold')
axS.legend(loc='lower right', framealpha=0.95)
axS.grid(True, alpha=0.3)

figS.savefig(os.path.join(FIG_DIR, 'figure_S1_calibration.pdf'))
plt.close(figS)
print("Saved: figure_S1_calibration.pdf")

print("\nAll figures generated successfully.")
