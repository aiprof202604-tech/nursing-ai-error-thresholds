"""
Round-2 robustness figure (Supplementary Figure S2) for JMIR Nursing #99590.

Panel A: joint human-AI decision model -- required AI accuracy (A_threshold)
         as a function of the clinician's independent accuracy p_c, for the
         highest-risk cell (novice x high complexity) and the lowest-deference
         cell (experienced x low complexity). Base model = p_c -> 1.
Panel B: global sensitivity -- distribution of the novice x high A_threshold
         (<10% target) under a wide theory-signed prior and a fully sign-
         agnostic prior over the four literature-fixed coefficients.

Outputs: figures/figure_S2_robustness.pdf  (vector)
Self-contained; mirrors constants from 03/05. Seed 42.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

B0, BA = 0.4708, 0.2010
BE, BC, BAE, BAC = -0.20, 0.20, 0.10, 0.10
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
A_GRID = np.linspace(0.001, 0.999, 1000)


def reliance(A, E, C):
    return np.clip(B0 + BA * A + BE * E + BC * C + BAE * A * E + BAC * A * C, 0, 1)


def joint_thr(E, C, p_c, target):
    err = reliance(A_GRID, E, C) * (1 - A_GRID) + (1 - reliance(A_GRID, E, C)) * (1 - p_c)
    below = np.where(err < target)[0]
    return float(A_GRID[below[0]]) if below.size else np.nan


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0))

# ---- Panel A: joint model across clinician accuracy p_c ----
pc = np.linspace(0.5, 1.0, 101)
cells = [("Novice x high complexity", 0.0, 1.0, "#C0392B"),
         ("Experienced x low complexity", 1.0, 0.0, "#2980B9")]
for lbl, E, C, col in cells:
    t10 = np.array([joint_thr(E, C, p, 0.10) for p in pc])
    t20 = np.array([joint_thr(E, C, p, 0.20) for p in pc])
    axA.plot(pc, t10, color=col, ls="-", lw=1.9, label=f"{lbl}  (<10% target)")
    axA.plot(pc, t20, color=col, ls="--", lw=1.7, label=f"{lbl}  (<20% target)")

axA.axvline(1.0, color="#555555", ls=":", lw=0.9)
axA.annotate("p_c = 1 recovers\nthe base model", xy=(1.0, 0.55),
             xytext=(0.72, 0.50), fontsize=8, color="#555555", style="italic",
             arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
axA.text(0.515, 0.965,
         "Lines ending before p_c = 0.5 indicate the target is\n"
         "unreachable at any AI accuracy (clinician-error floor)",
         fontsize=7.5, color="#8A5A00", style="italic", va="top")
axA.set_xlim(0.5, 1.0)
axA.set_ylim(0.45, 1.0)
axA.set_xlabel("Clinician independent accuracy $p_c$")
axA.set_ylabel(r"Required AI accuracy ($A_{threshold}$)")
axA.set_title("(A) Joint decision model: threshold vs clinician accuracy",
              fontweight="bold")
axA.legend(loc="lower left", framealpha=0.95, fontsize=7.3)
axA.grid(True, alpha=0.3)

# ---- Panel B: global sensitivity distributions (novice x high, <10%) ----
rng = np.random.default_rng(42)
n = 50000
Agr = A_GRID[None, :]
E, C = 0.0, 1.0


def mc_thr(bE, bC, bAE, bAC, target=0.10):
    r = np.clip(B0 + BA * Agr + bE[:, None] * E + bC[:, None] * C
                + bAE[:, None] * (Agr * E) + bAC[:, None] * (Agr * C), 0, 1)
    err = r * (1 - Agr)
    mask = err < target
    idx = np.argmax(mask, axis=1)
    thr = A_GRID[idx].astype(float)
    thr[~mask.any(axis=1)] = np.nan
    return thr[~np.isnan(thr)]


signed = mc_thr(rng.uniform(-0.4, 0, n), rng.uniform(0, 0.4, n),
                rng.uniform(0, 0.2, n), rng.uniform(0, 0.2, n))
agn = mc_thr(rng.uniform(-0.4, 0.4, n), rng.uniform(-0.4, 0.4, n),
             rng.uniform(-0.2, 0.2, n), rng.uniform(-0.2, 0.2, n))

bins = np.linspace(0.3, 1.0, 60)
axB.hist(signed, bins=bins, density=True, alpha=0.55, color="#2C7FB8",
         label="Wide prior, theory-consistent signs")
axB.hist(agn, bins=bins, density=True, alpha=0.5, color="#E67E22",
         label="Fully sign-agnostic prior")
axB.axvline(0.894, color="#C0392B", ls="-", lw=1.6,
            label="Reported estimate (0.894)")
axB.axvline(0.70, color="#555555", ls=":", lw=1.1,
            label="Substantial-accuracy anchor (0.70)")
axB.set_xlim(0.35, 1.0)
axB.set_xlabel(r"Novice $\times$ high-complexity $A_{threshold}$ (<10% target)")
axB.set_ylabel("Density")
axB.set_title("(B) Global sensitivity over all fixed coefficients",
              fontweight="bold")
axB.legend(loc="upper left", framealpha=0.95, fontsize=7.5)
axB.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "figure_S2_robustness.pdf"))
plt.close(fig)
print("Saved: figure_S2_robustness.pdf")
print(f"  signed prior: median={np.median(signed):.3f}, "
      f"P(>=0.70)={np.mean(signed>=0.70):.3f}")
print(f"  agnostic prior: median={np.median(agn):.3f}, "
      f"P(>=0.70)={np.mean(agn>=0.70):.3f}")
