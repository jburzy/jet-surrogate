"""Generate Pythia8 + Delphes samples (one ROOT file per seed).

    jet-surrogate generate --sample signal --ctau 0.1 --nevents 10000 --seed 1
    jet-surrogate generate --sample signal --grid --mpid 10 --nevents 10000
    jet-surrogate generate --sample qcd --nevents 10000 --seed 1
    jet-surrogate generate --sample signal --ctau 0.1 --lambda 25 --seed 1   # m_pi / Lambda scan point
    jet-surrogate generate --sample signal --ctau 0.1 --nflav 2 --seed 1     # dark-flavour scan point
"""

from __future__ import annotations

import time

from ..generate import CTAU_GRID_MM, NOMINAL_MPID, generate_hepmc, generate_sample


def add_arguments(ap) -> None:
    ap.add_argument("--sample", choices=["signal", "qcd"], default=None, help="built-in sample (or use --card)")
    ap.add_argument("--card", default=None, help="any Pythia .cmnd card (HepMC output only): e.g. cards/pythia/ggh_ss_bb.cmnd")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra Pythia setting appended to the card, e.g. --set '35:tau0 = 5320'")
    ap.add_argument("--ctau", type=float, default=None, help="dark-pion ctau in mm")
    ap.add_argument("--grid", action="store_true", help="run the full ctau grid (signal)")
    ap.add_argument("--mpid", type=float, default=NOMINAL_MPID,
                    help="dark-pion mass in GeV; the whole dark sector is rescaled coherently")
    ap.add_argument("--lambda", dest="lam", type=float, default=None,
                    help="HiddenValley:Lambda in GeV at fixed masses (default: 2 m_pid, the nominal scaling)")
    ap.add_argument("--nflav", type=int, default=None, help="number of dark-quark flavours (default 1)")
    ap.add_argument("--mzp", type=float, default=None, help="Z' mass in GeV (default 1500)")
    ap.add_argument("--nevents", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/delphes")
    ap.add_argument("--quiet", action="store_true", help="suppress Pythia/Delphes stdout")
    ap.add_argument("--hepmc2", action="store_true",
                    help="write HepMC2 ASCII instead of HepMC3 (input format of Delphes' hepmc2pileup)")
    ap.add_argument("--format", choices=["delphes", "hepmc"], default="delphes",
                    help="delphes: DelphesPythia8 ROOT (full chain); hepmc: standalone Pythia8 -> HepMC3 "
                         "(input for `jet-surrogate predict`)")


def run(args) -> None:
    if args.card:
        t0 = time.time()
        out = generate_hepmc("card", n_events=args.nevents, seed=args.seed, card=args.card, hepmc_version=2 if args.hepmc2 else 3,
                             settings=[x.replace("=", " = ", 1) if "=" in x and " = " not in x else x for x in args.set],
                             out_dir=args.out if args.out != "data/delphes" else "data/hepmc")
        print(f"wrote {out}  ({args.nevents} events, {time.time() - t0:.0f} s)", flush=True)
        return
    if args.sample is None:
        raise SystemExit("--sample or --card is required")
    if args.sample == "signal":
        if args.grid:
            ctaus = list(CTAU_GRID_MM)
        elif args.ctau is not None:
            ctaus = [args.ctau]
        else:
            raise SystemExit("--ctau (or --grid) required for signal")
    else:
        ctaus = [None]
    for ctau in ctaus:
        t0 = time.time()
        if args.format == "hepmc":
            out = generate_hepmc(args.sample, n_events=args.nevents, seed=args.seed, ctau_mm=ctau,
                                 mpid=args.mpid, lam=args.lam, nflav=args.nflav,
                                 out_dir=args.out if args.out != "data/delphes" else "data/hepmc")
        else:
            out = generate_sample(args.sample, n_events=args.nevents, seed=args.seed,
                                  ctau_mm=ctau, mpid=args.mpid, lam=args.lam, nflav=args.nflav,
                                  mzp=args.mzp, out_dir=args.out, quiet=args.quiet)
        print(f"wrote {out}  ({args.nevents} events, {time.time() - t0:.0f} s)", flush=True)
