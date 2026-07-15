from __future__ import annotations

import math
from typing import Mapping


def resolve_particle_mass(garment_config: Mapping, particle_count: int) -> float:
    """Resolve Isaac Sim's per-particle mass from a garment mass contract."""
    if particle_count <= 0:
        raise ValueError("garment mesh must contain at least one particle")
    total_mass = garment_config.get("total_mass")
    particle_mass = garment_config.get("particle_mass")
    if total_mass is not None and particle_mass is not None:
        raise ValueError("garment_config cannot define both total_mass and particle_mass")
    if total_mass is not None:
        total_mass = float(total_mass)
        if not math.isfinite(total_mass) or total_mass <= 0:
            raise ValueError("garment total_mass must be a positive finite value")
        return total_mass / particle_count
    particle_mass = float(1e-2 if particle_mass is None else particle_mass)
    if not math.isfinite(particle_mass) or particle_mass <= 0:
        raise ValueError("garment particle_mass must be a positive finite value")
    return particle_mass
