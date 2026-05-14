"""Command-line entry point."""

import json
import click

from .pipeline import run_reco, run_pipeline


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("hepmc_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--model", "-m",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to ONNX surrogate model. Omit to run reconstruction only.",
)
@click.option(
    "--threshold", "-t",
    default=0.5,
    show_default=True,
    help="Score threshold for jet acceptance (ignored in reco-only mode).",
)
@click.option(
    "--pt-cut",
    default=200.0,
    show_default=True,
    help="Minimum pT [GeV] for large-R (R=1.0) jets.",
)
@click.option(
    "--dr-match",
    default=1.4,
    show_default=True,
    help="dR cone used to associate truth particles to large-R jets.",
)
@click.option(
    "--max-events", "-n",
    default=None,
    type=int,
    help="Stop after this many events (default: all).",
)
@click.option(
    "--max-particles",
    default=200,
    show_default=True,
    help="Maximum particles per jet passed to the model (full pipeline only).",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help=(
        "Output file. "
        "Reco-only mode: saves a .npz of jet kinematics and particle features. "
        "Full pipeline: saves a JSON of acceptance results."
    ),
)
@click.option("--verbose", "-v", is_flag=True, help="Print progress every 100 events.")
def main(
    hepmc_file,
    model,
    threshold,
    pt_cut,
    dr_match,
    max_events,
    max_particles,
    output,
    verbose,
):
    """
    Run the jet surrogate analysis pipeline on HEPMC_FILE.

    Without --model: reconstructs jets and extracts per-particle features only.
    With --model:    also runs ONNX inference and reports two-jet acceptance.
    """
    click.echo(f"Input : {hepmc_file}")
    click.echo(f"Cuts  : R=1.0 jet pT > {pt_cut} GeV | dR match < {dr_match}")
    if max_events:
        click.echo(f"Events: first {max_events}")

    if model is None:
        click.echo("Mode  : reco-only (no model provided)")
        results = run_reco(
            hepmc_file=hepmc_file,
            large_r_pt_cut=pt_cut,
            dr_match=dr_match,
            max_events=max_events,
            max_particles=max_particles,
            output_npz=output,
            verbose=verbose,
        )
        click.echo("")
        click.echo(f"Events processed : {results['n_events']}")
        click.echo(f"Large-R jets     : {results['n_jets']}")
        if results["n_jets"] > 0:
            pt = results["jet_pt"]
            npart = results["jet_n_particles"]
            click.echo(f"Jet pT           : mean={pt.mean():.1f}  min={pt.min():.1f}  max={pt.max():.1f} GeV")
            click.echo(f"Particles/jet    : mean={npart.mean():.1f}  min={npart.min()}  max={npart.max()}")
            click.echo(f"Particle features: {len(results['feature_names'])} vars — {', '.join(results['feature_names'])}")
        if output:
            click.echo(f"\nOutput           : {output}")
            click.echo(f"  /jets         [n_jets]                  pt eta energy mass phi")
            click.echo(f"  /particles    [n_jets, {max_particles}]  {len(results['feature_names'])} features + valid mask")
    else:
        click.echo(f"Model : {model}")
        click.echo(f"Cuts  : score > {threshold}")
        results = run_pipeline(
            hepmc_file=hepmc_file,
            model_path=model,
            threshold=threshold,
            large_r_pt_cut=pt_cut,
            dr_match=dr_match,
            max_events=max_events,
            max_particles=max_particles,
            verbose=verbose,
        )
        click.echo("")
        click.echo(f"Events processed       : {results['n_events']}")
        click.echo(f"Events with >=2 tagged : {results['n_events_2jets']}")
        click.echo(f"Acceptance             : {results['acceptance']:.6f}")
        if output:
            payload = {k: v for k, v in results.items() if k != "all_scores"}
            with open(output, "w") as f:
                json.dump(payload, f, indent=2)
            click.echo(f"\nResults written to {output}")
