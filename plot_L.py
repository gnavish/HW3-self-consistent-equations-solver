#!/usr/bin/env python3

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1, 2, 3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
FLOOR = 1e-13          # solver tolerance is 1e-12 in the residual; deviations below this are noise
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "mathtext.fontset": "cm",
})

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("csv", nargs="?", default="sweep_L.csv", help="CSV file from sweep_L.cpp")
ap.add_argument("--out", default=None, help="output file stem (default: CSV name without .csv)")
ap.add_argument("--J", default=None, help="J1 = J2 value, only used in the title")
a = ap.parse_args()
stem = a.out or (a.csv[:-4] if a.csv.endswith(".csv") else a.csv)

try:
    d = np.loadtxt(a.csv, delimiter=",", skiprows=1, ndmin=2)
except OSError as e:
    sys.exit(f"cannot read {a.csv}: {e}")
if d.shape[0] < 2 or d.shape[1] != 5:
    sys.exit(f"{a.csv}: expected 5 columns (L,A1,A2,mu,gap) and at least two rows, got shape {d.shape}")
d = d[np.argsort(d[:, 0])]
L, A1, A2, mu, gap = d.T

quantities = [(A1, r"$A_1$", BLUE), (A2, r"$A_2$", ORANGE), (mu, r"$\mu$", AQUA)]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
panels = [axes[0, 0], axes[0, 1], axes[1, 0]]

# (a)-(c): raw values vs L, each on its own axis
for ax, tag, (y, name, color) in zip(panels, "abc", quantities):
    ax.axhline(y[-1], color=INK2, lw=1, ls="--", zorder=1)
    ax.plot(L, y, "-o", color=color, lw=2, ms=5, zorder=2)
    ax.annotate(rf"{name}$(L={L[-1]:.0f}) = {y[-1]:.8f}$", (L[-1], y[-1]), xytext=(-4, 10),
                textcoords="offset points", ha="right", va="bottom", color=INK, fontsize=10)
    ax.set_xlabel(r"$L$"); ax.set_ylabel(name)
    ax.set_title(f"({tag}) {name} vs system size", loc="left", color=INK, fontsize=11)
    pad = 0.15 * (y.max() - y.min() or abs(y[-1]) * 1e-3)
    ax.set_ylim(y.min() - pad, y.max() + 2 * pad)

# (d): relative deviation from the largest-L value, log scale
ax = axes[1, 1]
for y, name, color in quantities:
    dev = np.abs(y[:-1] - y[-1]) / abs(y[-1])
    keep = dev > FLOOR
    ax.semilogy(L[:-1][keep], dev[keep], "-o", color=color, lw=2, ms=5, label=name)
ax.axhline(FLOOR, color=INK2, lw=1, ls=":")
ax.text(L[0], FLOOR * 1.6, "solver tolerance", color=INK2, fontsize=9, va="bottom")
ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$|X(L) - X(L_{\max})|\ /\ X(L_{\max})$")
ax.set_title(r"(d) relative deviation from $L_{\max}$", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, loc="upper right")

for ax in axes.flat:
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0, L.max() * 1.04)

jtxt = f"$J_1 = J_2 = {a.J}$, " if a.J else ""
fig.suptitle(rf"Finite-size convergence of the $\pi$-flux $\mathbb{{Z}}_2$ mean-field solution "
             rf"({jtxt}$L_{{\max}} = {L[-1]:.0f}$)", color=INK, fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")

for y, name, _ in quantities:
    dev = np.abs(y[:-1] - y[-1]) / abs(y[-1])
    ok = L[:-1][dev < 1e-6]
    msg = f"converged to 1e-6 by L = {ok[0]:.0f}" if ok.size else "not converged to 1e-6"
    print(f"{name.strip('$'):>4}: {y[-1]:.10f}  ({msg})")
print(f"read {len(L)} sizes from {a.csv}; saved {stem}.png and {stem}.pdf")