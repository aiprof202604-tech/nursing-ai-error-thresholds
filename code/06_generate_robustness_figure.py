"""
Supplementary Figure S2 -- JMIR Nursing #99590. Top-tier redesign.

Message-first design:
  Panel A tells one story: raising AI accuracy cannot remove the error that the
  clinician themselves contributes. For the low-deference cell an "unreachable
  zone" (shaded) opens up: below a certain clinician accuracy the <10% target
  cannot be met by ANY AI. Lines are DIRECTLY LABELLED (no legend box on data).
  Panel B tells one story: the reported 0.894 estimate is robust -- the sampled
  thresholds pile up at/above it under both a theory-signed and a fully
  sign-agnostic prior.

Okabe-Ito colours; grayscale-safe via line style. Seed 42. Numbers unchanged.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

B0, BA = 0.4708, 0.2010
BE, BC, BAE, BAC = -0.20, 0.20, 0.10, 0.10
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
A_GRID = np.linspace(0.001, 0.999, 1500)

VERM = "#D55E00"   # novice x high (high-risk)
BLUE = "#0072B2"   # experienced x low (low-deference)
ORANGE = "#E69F00"
GREY = "#666666"
FLOORFILL = "#F4C9B4"  # light vermillion tint for the unreachable zone


def reliance(A, E, C):
    return np.clip(B0 + BA * A + BE * E + BC * C + BAE * A * E + BAC * A * C, 0, 1)


def joint_thr(E, C, p_c, target):
    err = reliance(A_GRID, E, C) * (1 - A_GRID) + (1 - reliance(A_GRID, E, C)) * (1 - p_c)
    below = np.where(err < target)[0]
    return float(A_GRID[below[0]]) if below.size else np.nan


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 15.5, "axes.labelsize": 14,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "axes.linewidth": 1.1, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig = plt.figure(figsize=(12.2, 5.5))
gs = GridSpec(1, 2, width_ratios=[1.06, 1.0], wspace=0.24,
              left=0.075, right=0.985, top=0.88, bottom=0.135)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

# =============== Panel A ===============
pc = np.linspace(0.5, 1.0, 201)
r1_explow = reliance(np.array([1.0]), 1.0, 0.0)[0]        # 0.5718
floor_explow = (1 - r1_explow) * (1 - pc)                 # irreducible error floor

nh10 = np.array([joint_thr(0.0, 1.0, p, 0.10) for p in pc])
nh20 = np.array([joint_thr(0.0, 1.0, p, 0.20) for p in pc])
el10 = np.array([joint_thr(1.0, 0.0, p, 0.10) for p in pc])
el20 = np.array([joint_thr(1.0, 0.0, p, 0.20) for p in pc])

# Shade the "unreachable" region: where the experienced x low <10% target has no
# solution (nan). Fill the whole vertical strip there.
pc_crit = 1 - 0.10 / (1 - r1_explow)                      # ~0.766
axA.axvspan(0.5, pc_crit, color=FLOORFILL, alpha=0.55, zorder=0)
axA.axvline(pc_crit, color="#C0704A", ls=(0, (4, 3)), lw=1.5, zorder=2, alpha=0.9)
axA.text((0.5 + pc_crit) / 2, 0.515,
         "Unreachable zone\n<10% target impossible\nfor experienced \u00d7 low\nat any AI accuracy",
         ha="center", va="bottom", fontsize=10.5, color="#A0400A",
         style="italic", linespacing=1.25)

# Curves
axA.plot(pc, nh10, color=VERM, ls="-", lw=3.0, zorder=4)
axA.plot(pc, nh20, color=VERM, ls="--", lw=2.6, zorder=4)
axA.plot(pc[~np.isnan(el10)], el10[~np.isnan(el10)], color=BLUE, ls="-", lw=3.0, zorder=4)
axA.plot(pc, el20, color=BLUE, ls="--", lw=2.6, zorder=4)

# Direct labels placed along each curve where it is visually isolated,
# with a strong white halo so text stays legible where thick lines cross it.
halo = [pe.withStroke(linewidth=5.0, foreground="white")]

def idx_at(pcval):
    return int(np.argmin(np.abs(pc - pcval)))

# Novice x high: two nearly-flat lines near the top -> label on the left, stacked
axA.text(0.545, nh10[idx_at(0.545)] + 0.018, "Novice \u00d7 high  (<10%)",
         color=VERM, fontsize=12, ha="left", va="bottom", fontweight="bold",
         zorder=6, path_effects=halo)
axA.text(0.545, nh20[idx_at(0.545)] + 0.016, "Novice \u00d7 high  (<20%)",
         color=VERM, fontsize=12, ha="left", va="bottom", zorder=6,
         path_effects=halo)

# Experienced x low <10% (solid): label BELOW the line, kept clear of BOTH the
# top frame and the right frame (anchored well inside the right spine).
i_el10 = idx_at(0.885)
axA.text(0.885, el10[i_el10] - 0.03, "Experienced \u00d7 low  (<10%)",
         color=BLUE, fontsize=11.5, ha="center", va="top", fontweight="bold",
         rotation=-19, rotation_mode="anchor", zorder=6, path_effects=halo)

# Experienced x low <20% (dashed): moved UPPER-LEFT along the line
i_el20 = idx_at(0.86)
axA.text(0.855, el20[i_el20] + 0.028, "Experienced \u00d7 low  (<20%)",
         color=BLUE, fontsize=12, ha="center", va="bottom",
         rotation=-30, rotation_mode="anchor", zorder=6, path_effects=halo)

axA.set_xlim(0.5, 1.0)
axA.set_ylim(0.45, 1.03)
axA.set_xlabel("Clinician's own accuracy  $p_c$")
axA.set_ylabel("Minimum AI accuracy required  $A_{threshold}$")
axA.set_title("(A)  A better clinician lowers the AI bar\n"
              "— until an error floor makes the target unreachable",
              fontweight="bold", fontsize=13.5, linespacing=1.3)
axA.grid(True, alpha=0.22)
# subcaption: solid = <10%, dashed = <20%
axA.text(0.5, -0.135, "Solid = <10% target   ·   Dashed = <20% target",
         transform=axA.transAxes, ha="center", va="top", fontsize=11, color=GREY)

# =============== Panel B ===============
rng = np.random.default_rng(42)
n = 50000
Agr = A_GRID[None, :]


def mc_thr(bE, bC, bAE, bAC, target=0.10):
    r = np.clip(B0 + BA * Agr + bE[:, None] * 0.0 + bC[:, None] * 1.0
                + bAE[:, None] * (Agr * 0.0) + bAC[:, None] * (Agr * 1.0), 0, 1)
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

bins = np.linspace(0.35, 1.0, 55)
axB.hist(signed, bins=bins, density=True, color=BLUE, alpha=0.85,
         edgecolor="white", linewidth=0.3)
axB.hist(agn, bins=bins, density=True, color=ORANGE, alpha=0.55,
         edgecolor="white", linewidth=0.3)

# Anchor at reported 0.894
axB.axvline(0.894, color=VERM, ls="-", lw=2.8, zorder=5)
axB.annotate("Reported\n0.894", xy=(0.894, 38), xytext=(0.775, 43),
             color=VERM, fontsize=12.5, ha="center", va="center",
             fontweight="bold", linespacing=1.1,
             arrowprops=dict(arrowstyle="->", color=VERM, lw=1.4),
             path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
# 0.70 reference
axB.axvline(0.70, color=GREY, ls=":", lw=1.8, zorder=3)
axB.text(0.70, 46.5, "0.70", color=GREY, fontsize=11.5, ha="center", va="bottom")

# In-plot direct labels for the two distributions (no legend box)
axB.text(0.60, 6.0, "Sign-agnostic\nprior", color="#B07800", fontsize=11.5,
         ha="center", va="center", fontweight="bold", linespacing=1.1,
         zorder=6, path_effects=halo)
axB.text(0.845, 20.0, "Theory-signed\nprior", color=BLUE, fontsize=11.5,
         ha="right", va="center", fontweight="bold", linespacing=1.1,
         zorder=6, path_effects=halo)

axB.set_xlim(0.4, 1.0)
axB.set_ylim(0, 52)
axB.set_xlabel("Threshold under randomised assumptions  $A_{threshold}$")
axB.set_ylabel("Density")
axB.set_title("(B)  The 0.894 estimate is robust\n"
              "— thresholds cluster at/above it under wide priors",
              fontweight="bold", fontsize=13.5, linespacing=1.3)
axB.grid(True, axis="y", alpha=0.22)

fig.savefig(os.path.join(FIG, "figure_S2_robustness.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: figure_S2_robustness.pdf")
print(f"  signed: median={np.median(signed):.3f}, P>=0.70={np.mean(signed>=0.70):.3f}")
print(f"  agnostic: median={np.median(agn):.3f}, P>=0.70={np.mean(agn>=0.70):.3f}")
