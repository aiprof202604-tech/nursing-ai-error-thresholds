"""
Figure 1 - conceptual model. Professional redesign for JMIR Nursing #99590.

Design system shared with Figure S2: Okabe-Ito accents; restrained "adult"
schematic (white cards, clean coloured borders + coloured titles, no candy
pastel fills); clear left-to-right causal flow inputs -> reliance -> error ->
outcome, with an explicit (1 - A) amplifier pathway; generous whitespace,
consistent typography, high data-ink ratio. No "acceptability" wording
(revision item 4): outcome reads "Error below target". Footnote N = 3,502.
Coefficients/numbers unchanged.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE = "#0072B2"; ORANGE = "#E69F00"; VERM = "#D55E00"; GREEN = "#009E73"
INK = "#1A1A1A"; GREY = "#5A5A5A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(10.6, 4.7))
ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.9); ax.axis("off")


def card(x, y, w, h, title, color, body=None, title_fs=12.5, body_fs=11,
         title_frac=0.68, body_frac=0.32):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.07",
                       linewidth=2.0, edgecolor=color, facecolor="white",
                       mutation_aspect=1.0, zorder=3)
    ax.add_patch(p)
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                color=color, fontsize=title_fs, fontweight="bold", zorder=4)
    else:
        ax.text(x + w / 2, y + h * title_frac, title, ha="center", va="center",
                color=color, fontsize=title_fs, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * body_frac, body, ha="center", va="center",
                color=INK, fontsize=body_fs, zorder=4)


# input cards (left) - widened so the longest label ("Clinician experience",
# measured 2.35 units at 11.5 pt) fits on one line with margin
ix, iw, ih = 0.25, 2.9, 0.92
CFS = 11.5
card(ix, 3.45, iw, ih, "AI accuracy", BLUE, body=r"$A$", title_fs=CFS, body_fs=12,
     title_frac=0.66, body_frac=0.30)
card(ix, 2.14, iw, ih, "Clinician experience", BLUE, body=r"$E$", title_fs=CFS,
     body_fs=12, title_frac=0.66, body_frac=0.30)
card(ix, 0.83, iw, ih, "Task complexity", BLUE, body=r"$C$", title_fs=CFS, body_fs=12,
     title_frac=0.66, body_frac=0.30)
icard_r = ix + iw  # 3.15

# reliance card (centre)
rx, ry, rw, rh = 3.6, 1.55, 3.75, 2.0
p = FancyBboxPatch((rx, ry), rw, rh, boxstyle="round,pad=0.02,rounding_size=0.07",
                   linewidth=2.0, edgecolor=ORANGE, facecolor="white", zorder=3)
ax.add_patch(p)
ax.text(rx + rw / 2, ry + rh * 0.80, "Reliance", ha="center", va="center",
        color=ORANGE, fontsize=14, fontweight="bold", zorder=4)
ax.text(rx + rw / 2, ry + rh * 0.46, r"$\beta_0 + \beta_A A + \beta_E E + \beta_C C$",
        ha="center", va="center", color=INK, fontsize=12.5, zorder=4)
ax.text(rx + rw / 2, ry + rh * 0.22, r"$+\ \beta_{AE}\,(A{\cdot}E) + \beta_{AC}\,(A{\cdot}C)$",
        ha="center", va="center", color=INK, fontsize=12.5, zorder=4)

# error card (right upper)
ex, ey, ew, eh = 7.85, 2.62, 2.55, 1.0
card(ex, ey, ew, eh, "Error", VERM, body=r"Reliance $\times\,(1-A)$",
     title_fs=12.5, body_fs=11.5, title_frac=0.66, body_frac=0.30)
ecx = ex + ew / 2  # 9.125

# outcome card (right lower)
ox, oy, ow, oh = 7.85, 0.98, 2.55, 1.0
card(ox, oy, ow, oh, "Error below target", GREEN, title_fs=12.5)

# arrows: inputs -> reliance
arr = dict(arrowstyle="-|>", mutation_scale=16, lw=1.9, color=GREY,
           shrinkA=1, shrinkB=2, zorder=2)
for cy in (3.91, 2.60, 1.29):
    ax.add_patch(FancyArrowPatch((icard_r, cy), (rx, 2.55), **arr))
# reliance -> error
ax.add_patch(FancyArrowPatch((rx + rw, 2.95), (ex, 3.12), arrowstyle="-|>",
             mutation_scale=16, lw=2.0, color=ORANGE, shrinkA=1, shrinkB=2, zorder=2))
# error -> outcome
ax.add_patch(FancyArrowPatch((ecx, ey), (ox + ow / 2, oy + oh),
             arrowstyle="-|>", mutation_scale=16, lw=2.0, color=GREEN,
             shrinkA=2, shrinkB=2, zorder=2))

# (1 - A) amplifier pathway: right-angle bus routed ABOVE the (taller) cards
BUS_Y = 4.55
bx = ix + iw - 0.35            # rises from the A card's top edge
ax.plot([bx, bx], [3.45 + ih, BUS_Y], color=VERM, lw=1.8, ls=(0, (5, 2)),
        zorder=1, solid_capstyle="round")
ax.plot([bx, ecx], [BUS_Y, BUS_Y], color=VERM, lw=1.8, ls=(0, (5, 2)),
        zorder=1, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((ecx, BUS_Y), (ecx, ey + eh + 0.02),
             arrowstyle="-|>", mutation_scale=15, lw=1.8, color=VERM,
             ls=(0, (5, 2)), shrinkA=0, shrinkB=1, zorder=1))
ax.text(6.0, BUS_Y, r"$(1-A)$", ha="center", va="center", fontsize=11.5,
        color=VERM, fontweight="bold", zorder=5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

# calibration footnote (centred under the whole schematic)
ax.text(5.3, 0.28,
        r"$\beta_0,\ \beta_A$ calibrated from 9 empirical reliance points "
        r"(Lu & Yin 2021; Yin et al. 2019; $N=3{,}502$).  "
        r"$\beta_E,\beta_C,\beta_{AE},\beta_{AC}$ fixed from the literature.",
        ha="center", va="center", fontsize=9.5, color=GREY, style="italic", zorder=4)

fig.savefig(os.path.join(FIG_DIR, "figure1_conceptual.pdf"))
plt.close(fig)
print("Saved: figure1_conceptual.pdf")
