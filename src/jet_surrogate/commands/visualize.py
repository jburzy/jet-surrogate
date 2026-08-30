"""Make every figure from stored outputs (no compute, no data access).

    jet-surrogate visualize                    # all figures into results/figures
    jet-surrogate visualize --only closure     # a subset: tagger | training | closure

Reads models/tagger/{roc.json, history.json, working_point.yaml},
models/surrogate/history.json and results/summary.json; writes PNG + PDF.
One figure per file (multi-panel layouts are assembled with subfigure in
LaTeX): tagger_roc, training_curves, sr_efficiency_m<M> (with a ratio
pad), jet_eff_vs_pt_m<M>, calibration_m<M>, surrogate_output_m<M>.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..generate import NOMINAL_MPID


def add_arguments(ap) -> None:
    ap.add_argument("--tagger", default="models/tagger")
    ap.add_argument("--surrogate", default="models/surrogate")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--only", choices=["tagger", "training", "closure", "shapes"], default=None)


def _mass_tag(m: float) -> str:
    return "nominal, test seeds" if m == NOMINAL_MPID else "unseen mass point"


def plot_tagger_roc(tagger: Path, out: Path) -> None:
    from ..plotting import color, decorate, plt, save
    roc = json.loads((tagger / "roc.json").read_text())
    wp = yaml.safe_load((tagger / "working_point.yaml").read_text())
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (ct, c) in enumerate(sorted(roc["curves"].items(), key=lambda kv: float(kv[0]))):
        es, eb = np.array(c["eff_sig"]), np.array(c["eff_bkg"])
        ok = eb > 0
        eff = wp["eff_signal"].get(ct, {}).get("eff")
        lab = f"c$\\tau$ = {float(ct):g} mm" + (f"  ($\\epsilon$ = {eff:.2f} at WP)" if eff is not None else "")
        ax.plot(es[ok], 1 / eb[ok], color=color(i), label=lab)
    ax.axhline(roc["rejection"], color="0.5", lw=0.8, ls="--")
    ax.text(0.02, roc["rejection"] * 1.15, "working point", fontsize=8, color="0.4")
    ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_ylim(0.5, 1e10)      # headroom for the annotation
    ax.set_xlabel("signal jet efficiency"); ax.set_ylabel("QCD jet rejection")
    decorate(ax, "track transformer tagger, test seeds")
    ax.legend(loc="lower left", fontsize=11)
    save(fig, out / "tagger_roc.png")


def plot_training(hist_paths: dict[str, Path], out: Path) -> None:
    """Training and validation BCE of both networks on one axis."""
    from ..plotting import color, decorate, plt, save
    fig, ax = plt.subplots(figsize=(7, 6))
    i = 0; lo, hi = float("inf"), 0.0
    for name, p in hist_paths.items():
        if not p.exists():
            continue
        h = json.loads(p.read_text())["history"]
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_loss"] for r in h], "--", color=color(i), label=f"{name}, training")
        ax.plot(ep, [r["val_loss"] for r in h], "-", color=color(i), label=f"{name}, validation")
        lo = min(lo, min(r["val_loss"] for r in h)); hi = max(hi, max(r["train_loss"] for r in h))
        i += 1
    if i == 0:
        plt.close(fig); return
    ax.set_xlabel("epoch"); ax.set_ylabel("binary cross-entropy")
    ax.set_ylim(0.9 * lo, hi + 1.4 * (hi - lo))
    decorate(ax)
    ax.legend(fontsize=11, loc="upper right")
    save(fig, out / "training_curves.png")


def plot_closure(results: dict, out: Path) -> None:
    """One axis per figure: SR efficiency and its ratio per mass point,
    per-jet efficiency vs pT per mass point, and one calibration plot."""
    from ..plotting import MARKERS, color, decorate, plt, save
    groups = defaultdict(dict)                  # mpid -> ctau -> result
    for r in results.values():
        if r["sample"] == "signal" and r.get("lam", -1) <= 0 and r.get("nflav", -1) <= 0:
            groups[r["mpid"]][r["ctau"]] = r
    sr_label = "SR: 2 jets, $p_T > 200$ GeV, tagger WP (1/1000)"

    for m in sorted(groups):
        cts = sorted(groups[m]); rs = [groups[m][c] for c in cts]
        x = np.arange(len(cts)); ticks = [f"{c:g}" for c in cts]
        a = np.array([r["sr_actual"] for r in rs]); ae = np.array([r["sr_actual_err"] for r in rs])
        p = np.array([r["sr_pred"] for r in rs]); pe = np.array([r["sr_pred_err"] for r in rs])
        mass = f"$m_{{\\pi_d}}$ = {m:g} GeV ({_mass_tag(m)})"

        # --- SR efficiency with the predicted / actual ratio pad
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 7.5), sharex=True, height_ratios=[3, 1.1],
                                      gridspec_kw={"hspace": 0.05})
        ax.errorbar(x - 0.08, a, ae, fmt="o", color=color(0), label="detector level")
        ax.errorbar(x + 0.08, p, pe, fmt="s", mfc="white", color=color(1), label="surrogate")
        ax.set_ylabel("signal-region efficiency")
        ax.set_ylim(0, max(1.0, 1.9 * float(np.nanmax(np.concatenate([a, p])))))
        decorate(ax, f"{sr_label}\n{mass}")
        ax.legend(loc="upper right", fontsize=11)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p / a; rerr = ratio * np.sqrt((pe / p) ** 2 + (ae / a) ** 2)
        axr.axhline(1, color="0.6", lw=0.8)
        axr.errorbar(x, ratio, rerr, fmt="s", mfc="white", color=color(1))
        axr.set_xticks(x, ticks); axr.set_xlim(-0.6, len(cts) - 0.4); axr.set_ylim(0.4, 1.6)
        axr.set_xlabel("dark-pion c$\\tau$ [mm]"); axr.set_ylabel("pred / actual")
        save(fig, out / f"sr_efficiency_m{m:g}.png")

        # --- per-jet efficiency vs truth jet pT
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, c in enumerate(cts):
            v = [b for b in groups[m][c]["vs_pt"] if b["hi"] <= 1000.0]
            if not v:
                continue
            xc = [(b["lo"] + b["hi"]) / 2 for b in v]
            ax.errorbar(xc, [b["actual"] for b in v], [b["actual_err"] for b in v], fmt=MARKERS[i],
                        color=color(i), label=f"c$\\tau$ = {c:g} mm")
            ax.plot(xc, [b["pred"] for b in v], "--", color=color(i))
        ax.set_xlim(150, 1000); ax.set_ylim(0, 1.8)
        ax.set_xlabel("truth large-R jet $p_T$ [GeV]"); ax.set_ylabel("per-jet pass probability")
        decorate(ax, mass)
        leg1 = ax.legend(loc="upper right", fontsize=11)
        ax.add_artist(leg1)
        # second legend: what the marker vs the dashed line mean, in black
        from matplotlib.legend_handler import HandlerTuple
        from matplotlib.lines import Line2D
        markers = tuple(Line2D([], [], color="black", marker=MARKERS[i], ls="none", ms=6) for i in range(len(cts)))
        dashed = Line2D([], [], color="black", ls="--")
        ax.legend([markers, dashed], ["detector level", "surrogate"], loc="upper right", fontsize=11,
                  bbox_to_anchor=(0.985, 0.80),
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, handlelength=3.5)
        save(fig, out / f"jet_eff_vs_pt_m{m:g}.png")

        # --- calibration: marker area proportional to the bin population
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8)
        for i, c in enumerate(cts):
            cal = groups[m][c]["calibration"]
            if not cal:
                continue
            n = np.array([b["n"] for b in cal], float); frac = n / n.sum()
            size = 20 + 380 * frac                              # points^2, 20 for empty bins, 400 for all jets
            ax.scatter([b["pred"] for b in cal], [b["actual"] for b in cal], s=size, marker=MARKERS[i],
                       color=color(i), alpha=0.85, label=f"c$\\tau$ = {c:g} mm", zorder=3)
            ax.errorbar([b["pred"] for b in cal], [b["actual"] for b in cal], [b["actual_err"] for b in cal],
                        fmt="none", ecolor=color(i), elinewidth=0.8, zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.6)
        ax.set_xlabel("surrogate probability"); ax.set_ylabel("observed pass fraction")
        decorate(ax, f"{mass}\nmarker area: fraction of jets in the bin")
        ax.legend(fontsize=11, loc="upper right")
        save(fig, out / f"calibration_m{m:g}.png")

        # --- surrogate output for tagged and untagged jets (if evaluate stored it)
        if all("prob_hist" in groups[m][c] for c in cts):
            fig, ax = plt.subplots(figsize=(7, 6))
            for i, c in enumerate(cts):
                hh = groups[m][c]["prob_hist"]; edges = np.array(hh["edges"])
                for key, ls, lab in (("tagged", "-", "tagged"), ("untagged", "--", "untagged")):
                    y = np.array(hh[key], float); y = y / max(y.sum(), 1) / np.diff(edges)
                    ax.stairs(y, edges, color=color(i), ls=ls,
                              label=f"c$\\tau$ = {c:g} mm, {lab}")
            ax.set_yscale("log"); ax.set_xlim(0, 1)
            ax.set_xlabel("surrogate probability"); ax.set_ylabel("normalized jets")
            ymax = ax.get_ylim()[1]; ax.set_ylim(ax.get_ylim()[0], ymax * 1e3)
            decorate(ax, mass)
            ax.legend(fontsize=9, loc="upper right", ncol=1)
            save(fig, out / f"surrogate_output_m{m:g}.png")


def plot_variants(results: dict, out: Path) -> None:
    """SR efficiency vs m_pi / Lambda and vs the number of dark flavours at the
    nominal mass and the scan lifetime, detector level vs surrogate. The
    nominal point (m_pi / Lambda = 0.5, nFlav = 1) is the nominal-mass test
    sample at the same lifetime."""
    from ..plotting import color, decorate, plt, save
    from ..generate import NOMINAL_LAMBDA, NOMINAL_MPID
    sr_label = "SR: 2 jets, $p_T > 200$ GeV, tagger WP (1/1000)"
    variants = [r for r in results.values() if r["sample"] == "signal" and (r.get("lam", -1) > 0 or r.get("nflav", -1) > 0)]
    if not variants:
        return
    ctau = variants[0]["ctau"]
    nominal = results.get(f"signal_m{NOMINAL_MPID:g}_ctau{ctau:g}mm")
    for kind, xlabel, key, xnom in (("lambda", "$m_{\\pi_d} / \\Lambda_d$", lambda r: NOMINAL_MPID / r["lam"], NOMINAL_MPID / NOMINAL_LAMBDA),
                                    ("nflav", "number of dark-quark flavours", lambda r: r["nflav"], 1)):
        pts = [r for r in variants if (r.get("lam", -1) > 0) == (kind == "lambda")]
        if not pts:
            continue
        xs = [key(r) for r in pts]; rs = list(pts)
        if nominal is not None:
            xs.append(xnom); rs.append(nominal)
        order = np.argsort(xs); xs = np.array(xs)[order]; rs = [rs[i] for i in order]
        a = np.array([r["sr_actual"] for r in rs]); ae = np.array([r["sr_actual_err"] for r in rs])
        p_ = np.array([r["sr_pred"] for r in rs]); pe = np.array([r["sr_pred_err"] for r in rs])
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 7.5), sharex=True, height_ratios=[3, 1.1],
                                      gridspec_kw={"hspace": 0.05})
        ax.errorbar(xs, a, ae, fmt="o", color=color(0), label="detector level")
        ax.errorbar(xs, p_, pe, fmt="s", mfc="white", color=color(1), label="surrogate")
        ax.set_ylabel("signal-region efficiency"); ax.set_ylim(0, max(1.0, 1.9 * float(np.nanmax(np.concatenate([a, p_])))))
        decorate(ax, f"{sr_label}\n$m_{{\\pi_d}}$ = {NOMINAL_MPID:g} GeV, c$\\tau$ = {ctau:g} mm (surrogate trained at the nominal point)")
        ax.legend(loc="upper right", fontsize=11)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p_ / a; rerr = ratio * np.sqrt((pe / p_) ** 2 + (ae / a) ** 2)
        axr.axhline(1, color="0.6", lw=0.8)
        axr.errorbar(xs, ratio, rerr, fmt="s", mfc="white", color=color(1))
        axr.set_ylim(0.4, 1.6); axr.set_xlabel(xlabel); axr.set_ylabel("pred / actual")
        if kind == "nflav":
            axr.set_xticks(sorted(set(int(x) for x in xs)))
        save(fig, out / f"sr_efficiency_scan_{kind}.png")


SHAPE_LABELS = {
    "trk_n": ("tracks per jet", False),
    "trk_n_displaced": ("displaced tracks per jet ($|d_0|/\\sigma > 3$)", False),
    "trk_frac_displaced": ("displaced-track fraction", False),
    "trk_d0sig": ("track $|d_0|/\\sigma_{d_0}$", True),
    "trk_absd0": ("track $|d_0|$ [mm]", True),
    "trk_pt": ("track $p_T$ [GeV]", True),
    "reco_logit": ("tagger logit", False),
    "reco_jet_mass": ("reco jet mass [GeV]", False),
    "part_n": ("truth particles per jet", False),
    "part_n_charged": ("stable charged particles per jet", False),
    "dark_n": ("dark hadrons per jet", False),
    "dark_pt": ("dark-hadron $p_T$ [GeV]", True),
    "dark_ptfrac": ("dark-hadron $p_T$ fraction of jet", False),
    "dark_decay_len": ("dark-hadron decay length [mm]", True),
    "sm_decayed_n": ("decayed SM hadrons per jet", False),
    "truth_jet_mass": ("truth jet mass [GeV]", False),
}
EFF_LABELS = {
    "reco_eff_vs_n_displaced": ("displaced tracks per jet ($|d_0|/\\sigma > 3$)", "tagger efficiency (reco jets)"),
    "reco_eff_vs_frac_displaced": ("displaced-track fraction", "tagger efficiency (reco jets)"),
    "truth_eff_vs_dark_n": ("dark hadrons per jet", "tagger efficiency (truth jets)"),
    "truth_eff_vs_part_n": ("truth particles per jet", "tagger efficiency (truth jets)"),
}


def plot_shapes(shapes: dict, out: Path) -> None:
    """One figure per observable per scan: normalized distributions (or
    efficiencies), one curve per variant, nominal always included."""
    from ..generate import NOMINAL_LAMBDA, NOMINAL_MPID
    from ..plotting import color, decorate, plt, save
    scans = {
        "lambda": sorted((r for r in shapes.values() if r["lam"] > 0 or (r["lam"] < 0 <= 1 and r["nflav"] < 0)),
                         key=lambda r: NOMINAL_MPID / (r["lam"] if r["lam"] > 0 else NOMINAL_LAMBDA)),
        "nflav": sorted((r for r in shapes.values() if r["nflav"] > 0 or (r["lam"] < 0 and r["nflav"] < 0)),
                        key=lambda r: max(r["nflav"], 1)),
    }
    def curve_label(kind, r):
        if kind == "lambda":
            return f"$m_{{\pi_d}}/\Lambda_d$ = {NOMINAL_MPID / (r['lam'] if r['lam'] > 0 else NOMINAL_LAMBDA):g}"                    + (" (nominal)" if r["lam"] < 0 else "")
        return f"$N_{{flav}}$ = {max(r['nflav'], 1)}" + (" (nominal)" if r["nflav"] < 0 else "")
    extra = "$m_{\pi_d}$ = 5 GeV, c$\tau$ = 0.1 mm, jets $p_T > 200$ GeV"
    for kind, rows in scans.items():
        for var, (xlabel, logx) in SHAPE_LABELS.items():
            fig, ax = plt.subplots(figsize=(7, 6))
            for i, r in enumerate(rows):
                h = r["hists"][var]
                e, d = np.array(h["edges"]), np.array(h["density"])
                ax.stairs(d, e, color=color(i), lw=1.6, label=curve_label(kind, r))
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel); ax.set_ylabel("normalized to unit area")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
            decorate(ax, extra)
            ax.legend(loc="upper right", fontsize=10)
            save(fig, out / f"shape_{var}_{kind}.png")
        for var, (xlabel, ylabel) in EFF_LABELS.items():
            fig, ax = plt.subplots(figsize=(7, 6))
            for i, r in enumerate(rows):
                rows_e = r["effs"][var]
                x = np.array([(b["lo"] + b["hi"]) / 2 for b in rows_e])
                y = np.array([b["eff"] for b in rows_e]); ye = np.array([b["err"] for b in rows_e])
                ax.errorbar(x, y, ye, fmt="o-", ms=3.5, lw=1.2, color=color(i), label=curve_label(kind, r))
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_ylim(0, 1.55)
            decorate(ax, extra)
            ax.legend(loc="upper right", fontsize=10)
            save(fig, out / f"{var}_{kind}.png")


def run(args) -> None:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tagger, surrogate, results = Path(args.tagger), Path(args.surrogate), Path(args.results)
    made = []
    if args.only in (None, "tagger") and (tagger / "roc.json").exists():
        plot_tagger_roc(tagger, out); made.append("tagger_roc")
    if args.only in (None, "training"):
        plot_training({"tagger": tagger / "history.json", "surrogate": surrogate / "history.json"}, out)
        made.append("training_curves")
    if args.only in (None, "closure") and (results / "summary.json").exists():
        res = json.loads((results / "summary.json").read_text())
        plot_closure(res, out); made.append("closure")
        plot_variants(res, out); made.append("scans")
    if args.only in (None, "shapes") and (results / "shapes.json").exists():
        plot_shapes(json.loads((results / "shapes.json").read_text()), out); made.append("shapes")
    print(f"figures in {out}: {', '.join(made) or 'nothing to plot yet'}")
