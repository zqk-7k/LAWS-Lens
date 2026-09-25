#!/usr/bin/env python3
"""Physics-forward redesign draft for Fig. 1 (workflow schematic).

Design goal (review comment 5): foreground the four actual contributions
rather than a generic AI pipeline:
  [1] waveform embedding similarity matrix (with a highlighted true-pair cell)
  [2] population time-delay prior (lensed vs background log-delay curves)
  [3] HEALPix sky-posterior overlap (two credible regions on a sky ellipse)
  [4] catalog ranking -> fixed-budget shortlist funnel -> Bayesian confirmation
Honesty footer retained. Vector output; team may restyle.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

TEAL, ORANGE, BLUE, RED, GREY = "#2a7f8e", "#e8a33d", "#3a6ea5", "#c1502e", "#8a94a6"
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})

fig = plt.figure(figsize=(7.08, 3.55))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
ax.axis("off")

def box(x, y, w, h, label, fc="#f4f6f8", ec=GREY, fs=6.6, weight="normal", tc="#222"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w/2, y + h + 1.0, label, ha="center", va="bottom",
            fontsize=fs, fontweight=weight, color=tc)

def arrow(x0, y0, x1, y1, c=GREY, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=9, color=c, lw=lw,
                                 shrinkA=2, shrinkB=2))

# ---------------- input: catalog of N events (strains) --------------------
bx, by, bw, bh = 2.5, 30, 13, 13
box(bx, by, bw, bh, "Gravitational-wave catalog\n$N$ events, $N(N\\!-\\!1)/2$ pairs",
    fs=6.4, weight="bold")
rng = np.random.default_rng(3)
for k, yy in enumerate(np.linspace(by + 2.4, by + bh - 2.4, 4)):
    t = np.linspace(0, 1, 220)
    f = 5 + 24 * t**2.6
    w = np.sin(2*np.pi*f*t) * t**1.4 * np.exp(-((t-1)*5)**2*0.0) \
        + 0.16*rng.standard_normal(t.size)
    ax.plot(bx + 1.2 + t*(bw - 2.4), yy + 1.25*w, lw=0.55,
            color=BLUE, alpha=0.9)

# ---------------- [1] waveform embedding similarity matrix ----------------
mx, my, mw, mh = 24, 30.5, 12.5, 12.5
M = rng.uniform(0.05, 0.5, (7, 7)); M = (M + M.T)/2; np.fill_diagonal(M, 1)
M[1, 5] = M[5, 1] = 0.97
ax.imshow(M, extent=(mx, mx+mw, my, my+mh), cmap="Blues", vmin=0, vmax=1,
          origin="lower", zorder=2)
ii, jj = 5, 1
cw = mw/7
ax.add_patch(plt.Rectangle((mx + jj*cw, my + ii*cw), cw, cw, fill=False,
                           ec=RED, lw=1.5, zorder=4))
ax.text(mx + mw/2, my + mh + 1.0,
        "1  Waveform-embedding\nsimilarity (InceptionTime)", ha="center",
        va="bottom", fontsize=6.6, fontweight="bold", color="#222")
ax.annotate("true image pair", xy=(mx + (jj+0.5)*cw, my + (ii+0.5)*cw),
            xytext=(mx + mw + 1.2, my + mh - 1.2), fontsize=5.4, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
arrow(bx + bw + 1.2, by + bh/2, mx - 1.2, my + mh/2)
ax.text((bx+bw+mx)/2, by + bh/2 + 1.6, "encode +\nANN search", ha="center",
        fontsize=5.4, color=GREY)

# ---------------- [2] population time-delay prior -------------------------
tx, ty, tw, th = 24, 6, 12.5, 13
axt = fig.add_axes([tx/100, ty/50, tw/100, th/50])
x = np.linspace(-1.5, 3.2, 200)
lensed = np.exp(-0.5*((x-1.4)/0.65)**2)
bg = np.exp(-0.5*((x-2.55)/0.5)**2)*0.9
axt.plot(x, lensed, color=TEAL, lw=1.4)
axt.plot(x, bg, color=GREY, lw=1.2, ls="--")
axt.fill_between(x, lensed, color=TEAL, alpha=0.15)
axt.text(0.06, 0.86, "lensed\n(GW-LMC)", transform=axt.transAxes, fontsize=5.2,
         color=TEAL)
axt.text(0.62, 0.86, "unrelated\npairs", transform=axt.transAxes, fontsize=5.2,
         color=GREY)
axt.set_xticks([]); axt.set_yticks([])
axt.set_xlabel(r"$\log_{10}\Delta t$", fontsize=5.6, labelpad=1)
for s in axt.spines.values(): s.set_color(GREY)
ax.text(tx + tw/2, ty + th + 1.0, "2  Population time-delay prior",
        ha="center", va="bottom", fontsize=6.6, fontweight="bold")

# ---------------- [3] HEALPix sky overlap ---------------------------------
sx, sy, sw, sh = 41, 6, 13, 13
ax.add_patch(Ellipse((sx + sw/2, sy + sh/2 - 0.5), sw, sh*0.72, fc="#eef2f6",
                     ec=GREY, lw=0.9))
ax.add_patch(Ellipse((sx + sw/2 - 1.6, sy + sh/2 - 0.2), 4.6, 3.1, fc=BLUE,
                     alpha=0.45, ec="none"))
ax.add_patch(Ellipse((sx + sw/2 + 1.4, sy + sh/2 - 0.9), 5.2, 3.4, fc=ORANGE,
                     alpha=0.45, ec="none"))
ax.text(sx + sw/2, sy + sh + 1.0,
        "3  HEALPix sky-posterior\noverlap", ha="center", va="bottom",
        fontsize=6.6, fontweight="bold")
ax.text(sx + sw/2, sy - 0.6, r"$O^{\rm sky}_{ij}$ (cosine)", ha="center",
        fontsize=5.4, color=GREY)

# ---------------- fusion node ---------------------------------------------
fx, fy, fw, fh = 45, 31.5, 15, 10
box(fx, fy, fw, fh, "", fc="#fdf6ec", ec=ORANGE)
ax.text(fx + fw/2, fy + fh - 2.0,
        "Catalog ranking score", ha="center", fontsize=6.8, fontweight="bold")
ax.text(fx + fw/2, fy + fh/2 - 1.4,
        r"$S_{ij}=\lambda_{\rm w}z_{\rm w}+\lambda_{\rm t}z_{\rm t}"
        r"+\lambda_{\rm s}z_{\rm s}$" + "\nweights validation-selected",
        ha="center", fontsize=5.8)
arrow(mx + mw/2 + 3.5, my - 0.8, fx + 2.5, fy + fh + 0.6, c=BLUE)
arrow(tx + tw - 0.5, ty + th + 1.8, fx + 3.5, fy - 0.8, c=TEAL)
arrow(sx + sw/2, sy + sh + 3.0, fx + fw/2 + 1.5, fy - 0.8, c=ORANGE)

# ---------------- [4] fixed-budget shortlist funnel ------------------------
ux, uy = 68, 27
levels = [(r"all pairs  $\sim N^2/2$", 26, GREY),
          ("catalog-ranked candidates", 17, TEAL),
          ("fixed follow-up budget\n(top 10--50)", 9.5, RED)]
yy = uy + 14
for lab, w, c in levels:
    ax.add_patch(plt.Rectangle((ux + (26 - w)/2, yy), w, 4.2, fc=c, alpha=0.28,
                               ec=c, lw=1.0))
    ax.text(ux + 13, yy + 2.1, lab, ha="center", va="center", fontsize=5.6)
    yy -= 6.6
ax.text(ux + 13, uy + 19.6, "4  Fixed-budget shortlist", ha="center",
        va="bottom", fontsize=6.6, fontweight="bold")
arrow(fx + fw + 1.0, fy + fh/2, ux + 2.0, uy + 15.5)

# ---------------- endpoint: Bayesian confirmation --------------------------
ex, ey, ew, eh = 70.5, 4.5, 21, 7
box(ex, ey, ew, eh, "", fc="#f2f7f2", ec="#5f9e6e")
ax.text(ex + ew/2, ey + eh/2 + 1.2, "Bayesian confirmation", ha="center",
        fontsize=6.6, fontweight="bold", color="#2f6b3c")
ax.text(ex + ew/2, ey + eh/2 - 1.6,
        "posterior overlap / joint PE / phase consistency",
        ha="center", fontsize=5.4, color="#2f6b3c")
arrow(ux + 13, uy + 1.2, ex + ew/2, ey + eh + 0.8, c=RED)

ax.text(50, 1.0,
        "shortlists are ranked catalog coincidences that feed Bayesian confirmation; they are not lensing detections",
        ha="center", fontsize=5.6, color="#888888", style="italic")

fig.savefig("/home/claude/nc_v9/figures/workflow_physics_draft.pdf")
print("saved workflow_physics_draft.pdf")
