"""Shared plotting conventions: mplhep ATLAS style (as in displaced-observables),
a fixed colorblind-safe categorical palette, and the standard annotation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402

plt.style.use(hep.style.ATLAS)

PYTHIA_VERSION = "8.312"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]   # Okabe-Ito
MARKERS = ["o", "s", "^", "D", "v", "P"]


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def decorate(ax, extra: str | None = None, loc=(0.04, 0.96), fontsize=12) -> None:
    """ATLAS-style annotation (no experiment label): generator, CoM energy,
    the BSM process and the jet algorithm, plus an optional extra line."""
    lines = [
        f"Pythia {PYTHIA_VERSION} + Delphes, $\\sqrt{{s}} = 13.6$ TeV",
        "$Z'(1.5\\,\\mathrm{TeV}) \\to q_{\\mathrm{D}}\\bar{q}_{\\mathrm{D}}$",
        "anti-$k_t$ $R=1.0$ reclustered from $R=0.4$",
    ]
    if extra:
        lines.append(extra)
    ax.text(*loc, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=fontsize)


def save(fig, path) -> None:
    """PNG for inspection and PDF for the paper, side by side."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
