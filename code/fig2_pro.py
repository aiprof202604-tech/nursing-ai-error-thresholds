"""
Figure 2 - professional redesign for JMIR Nursing #99590.
Panel A: predicted error vs AI accuracy for 3 experience levels (high complexity).
Panel B: required AI accuracy (A_threshold) vs target error, 5 profiles, bootstrap 95% CI.

Key change vs the deployed version: NO legend boxes. Every curve carries a
direct, colour-matched label (with a white halo and high z-order so it stays
legible where lines cross), so there is never ambiguity about which line is which.
Okabe-Ito palette; line style also encodes the series (grayscale-safe).
Numbers unchanged; bootstrap loaded from figures/bootstrap_results.npz (seed 42).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
BE, BC, BAE, BAC = -0.20, 0.20, 0.10, 0.10
B0, BA = 0.4708, 0.2010
VERM = "#D55E00"; ORANGE = "#E69F00"; GREEN = "#009E73"
BLUE = "#0072B2"; PURPLE = "#CC79A7"; GREY = "#5A5A5A"
halo = [pe.withStroke(linewidth=4.5, foreground="white")]

def rel(A, E, C):
    return np.clip(B0 + BA*A + BE*E + BC*C + BAE*A*E + BAC*A*C, 0, 1)

def athr(t, E, C, b0=B0, bA=BA):
    Ag = np.linspace(0.001, 0.999, 1000)
    e = np.clip(b0 + bA*Ag + BE*E + BC*C + BAE*Ag*E + BAC*Ag*C, 0, 1) * (1 - Ag)
    bl = np.where(e <= t)[0]
    return Ag[bl[0]] if len(bl) else np.nan

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12.5,
    "axes.titlesize": 14.5, "axes.labelsize": 13.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

bd = np.load(os.path.join(FIG_DIR, "bootstrap_results.npz"))
boot_b0, boot_bA = bd["boot_b0"], bd["boot_bA"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.3))

# =============================== Panel A ===============================
A = np.linspace(0.0, 1.0, 401)
# (E, name, colour, linestyle, label_x, dy)  -- dy: vertical offset of label from curve
seriesA = [
    (0.0, "Novice (0 yr)",        VERM,   "-",   0.30,  0.055),
    (0.2, "Junior (2 yr)",        ORANGE, "--",  0.52, -0.050),
    (1.0, "Experienced (10 yr)",  GREEN,  "-.",  0.66, -0.052),
]
for E, name, col, ls, lx, dy in seriesA:
    err = rel(A, E, 1.0) * (1 - A)
    axA.plot(A, err, color=col, ls=ls, lw=2.6, zorder=4)
    i = int(np.argmin(np.abs(A - lx)))
    # local slope -> rotation (in display space)
    axA.annotate(name, xy=(A[i], err[i]), xytext=(A[i], err[i] + dy),
                 color=col, fontsize=12, fontweight="bold", ha="center",
                 va="center", zorder=6, path_effects=halo)

# current general-purpose LLM accuracy band (reference region, not a data line)
axA.axvspan(0.5, 0.7, color="#9ECAE1", alpha=0.35, zorder=0)
axA.text(0.60, 0.575, "Current general-purpose\nLLM range (complex tasks)",
         fontsize=10.5, color="#20567F", ha="center", va="top", style="italic",
         zorder=6, path_effects=halo)

# error reference lines
for y, lab in [(0.30, "30% error"), (0.20, "20% error"), (0.10, "10% error")]:
    axA.axhline(y, color=GREY, ls=":", lw=1.0, alpha=0.6, zorder=1)
    axA.text(0.985, y + 0.008, lab, fontsize=10.5, color=GREY, ha="right",
             va="bottom", zorder=6, path_effects=halo)

axA.set_xlim(0, 1); axA.set_ylim(0, 0.60)
axA.set_xlabel("AI accuracy  $A$")
axA.set_ylabel("Predicted error rate")
axA.set_title("(A)  Error falls as AI accuracy rises", fontweight="bold")
axA.grid(True, alpha=0.22)

# =============================== Panel B ===============================
err_grid = np.linspace(0.05, 0.30, 51)
# (name, E, C, colour, ls, mode, anchor)  mode: 'end' right-end label / 'curve' on-curve
seriesB = [
    ("Novice \u00d7 High",       0.0, 1.0, VERM,   "-",   "end",   0.655),
    ("Junior \u00d7 High",       0.2, 1.0, PURPLE, "--",  "end",   0.605),
    ("Novice \u00d7 Moderate",   0.0, 0.5, ORANGE, "--",  "end",   0.555),
    ("Experienced \u00d7 High",  1.0, 1.0, GREEN,  "-.",  "end",   0.505),
    ("Experienced \u00d7 Low",   1.0, 0.0, BLUE,   ":",   "curve", None),
]
x100 = err_grid * 100
for name, E, C, col, ls, mode, anchor in seriesB:
    a_pt = np.array([athr(t, E, C) for t in err_grid])
    boot = np.empty((len(boot_bA), len(err_grid)))
    for i, (b0i, bAi) in enumerate(zip(boot_b0, boot_bA)):
        boot[i] = [athr(t, E, C, b0i, bAi) for t in err_grid]
    lo = np.nanpercentile(boot, 2.5, axis=0); hi = np.nanpercentile(boot, 97.5, axis=0)
    axB.fill_between(x100, lo, hi, color=col, alpha=0.15, linewidth=0, zorder=2)
    axB.plot(x100, a_pt, color=col, ls=ls, lw=2.7, zorder=4)
    if mode == "end":
        y_end = a_pt[-1]                      # value at 30%
        axB.annotate(name, xy=(30, y_end), xytext=(30.7, anchor),
                     color=col, fontsize=11.5, fontweight="bold", ha="left",
                     va="center", zorder=6, path_effects=halo,
                     arrowprops=dict(arrowstyle="-", color=col, lw=1.0,
                                     shrinkA=1, shrinkB=1))
    else:
        # label the steep Experienced x Low curve directly, mid-curve where isolated
        j = int(np.argmin(np.abs(x100 - 15)))
        axB.annotate(name, xy=(x100[j], a_pt[j]), xytext=(x100[j] - 0.3, a_pt[j] + 0.075),
                     color=col, fontsize=11.5, fontweight="bold", ha="center",
                     va="center", zorder=6, path_effects=halo)

for y in (0.95, 0.85, 0.70):
    axB.axhline(y, color="#999999", ls=":", lw=0.9, alpha=0.6, zorder=1)
    axB.text(5.2, y + 0.006, f"{y:.2f}", fontsize=10, color=GREY, ha="left",
             va="bottom", zorder=6, path_effects=halo)

axB.set_xlim(5, 34.5); axB.set_xticks([5, 10, 15, 20, 25, 30])
axB.set_ylim(0.40, 1.00)
axB.set_xlabel("Target error rate (%)")
axB.set_ylabel("Required AI accuracy  $A_{threshold}$")
axB.set_title("(B)  Stricter targets demand more accurate AI", fontweight="bold")
axB.grid(True, alpha=0.22)

fig.tight_layout(pad=1.3, w_pad=2.4)
fig.savefig(os.path.join(FIG_DIR, "figure2_predicted_and_threshold.pdf"))
plt.close(fig)
print("Saved: figure2_predicted_and_threshold.pdf")
