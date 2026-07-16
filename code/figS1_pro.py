"""
Supplementary Figure S1 - calibration fit. Professional redesign for #99590.
Scatter of the 9 empirical reliance points from 3 studies vs the calibrated
linear model (E=0, C=0.5), with a bootstrap 95% band on the model line.

Design system shared with Figure S2: Okabe-Ito palette, larger fonts, subtle
grid, markers lifted off the grid with a white edge, message-driven title,
bootstrap uncertainty shown as a shaded band. N = 3,502; RMSE = 0.054.
Numbers unchanged; bootstrap loaded from figures/bootstrap_results.npz (seed 42).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
BE, BC, BAE, BAC = -0.20, 0.20, 0.10, 0.10
B0, BA = 0.4708, 0.2010
BLUE = "#0072B2"; ORANGE = "#E69F00"; GREEN = "#009E73"; INK = "#333333"; GREY = "#5A5A5A"
halo = [pe.withStroke(linewidth=4.0, foreground="white")]

STUDIES = {
    "Lu&Yin_Exp2":   [(0.50, 0.642), (0.80, 0.664)],
    "Yin19_Exp3_P2": [(0.55, 0.795), (0.80, 0.830), (1.00, 0.833)],
    "Yin19_Exp1_P1": [(0.60, 0.745), (0.70, 0.760), (0.90, 0.800), (0.95, 0.820)],
}
STYLE = {
    "Lu&Yin_Exp2":   (BLUE,   "o", "Lu & Yin 2021, Exp 2",            466),
    "Yin19_Exp3_P2": (ORANGE, "s", "Yin et al. 2019, Exp 3 (Phase 2)", 1042),
    "Yin19_Exp1_P1": (GREEN,  "^", "Yin et al. 2019, Exp 1 (Phase 1)", 1994),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12.5,
    "axes.titlesize": 14.5, "axes.labelsize": 13.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

bd = np.load(os.path.join(FIG_DIR, "bootstrap_results.npz"))
boot_b0, boot_bA = bd["boot_b0"], bd["boot_bA"]
rmse = float(bd["rmse_point"])

fig, ax = plt.subplots(figsize=(9.2, 6.2))

# calibrated model line (E=0, C=0.5)
A = np.linspace(0.45, 1.0, 200)
line = B0 + BA * A + BC * 0.5
ax.plot(A, line, color=INK, lw=2.6, zorder=2,
        label=r"Calibrated model ($E=0,\ C=0.5$)")

# empirical points
for s, pts in STUDIES.items():
    col, mk, lbl, n = STYLE[s]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.scatter(xs, ys, s=155, c=col, marker=mk, edgecolors="white",
               linewidths=1.4, zorder=4, label=f"{lbl}  ($N={n:,}$)")

# fit-quality annotation (accurate: scatter reflects between-study baseline differences)
ax.text(0.47, 0.870,
        f"RMSE = {rmse:.3f} over 9 points\nscatter reflects between-study\nbaseline reliance differences",
        fontsize=11, color=GREY, ha="left", va="top", style="italic",
        zorder=6, path_effects=halo)

ax.set_xlim(0.45, 1.02)
ax.set_ylim(0.60, 0.90)
ax.set_xlabel("AI accuracy  $A$")
ax.set_ylabel("Reliance (agreement fraction)")
ax.set_title("Calibrated reliance model vs 9 empirical points from 3 studies\n"
             r"($N = 3{,}502$)", fontweight="bold")
ax.grid(True, alpha=0.22)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
          fontsize=11.5, columnspacing=1.8, handletextpad=0.6, labelspacing=0.7)

fig.savefig(os.path.join(FIG_DIR, "figure_S1_calibration.pdf"))
plt.close(fig)
print("Saved: figure_S1_calibration.pdf  (RMSE=%.4f)" % rmse)
