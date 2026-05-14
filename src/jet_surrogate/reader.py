"""HEPMC file reading. Supports both HEPMC2 and HEPMC3 formats via pyhepmc."""

from typing import Generator
import pyhepmc


def iter_events(filepath: str) -> Generator:
    """Yield GenEvent objects from a HEPMC file."""
    with pyhepmc.open(filepath) as f:
        for event in f:
            yield event
