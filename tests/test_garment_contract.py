import pytest

from robodojo.sim.environment.scene_manager.garment_contract import resolve_particle_mass


def test_total_garment_mass_is_distributed_across_mesh_particles():
    assert resolve_particle_mass({"total_mass": 0.2}, 10_000) == pytest.approx(2e-5)


def test_legacy_per_particle_mass_remains_supported():
    assert resolve_particle_mass({"particle_mass": 3e-5}, 10_000) == pytest.approx(3e-5)
    assert resolve_particle_mass({}, 10_000) == pytest.approx(1e-2)


@pytest.mark.parametrize(
    ("config", "particle_count"),
    [
        ({"total_mass": 0.2, "particle_mass": 2e-5}, 10_000),
        ({"total_mass": 0.0}, 10_000),
        ({"particle_mass": float("nan")}, 10_000),
        ({"total_mass": 0.2}, 0),
    ],
)
def test_invalid_garment_mass_contracts_are_rejected(config, particle_count):
    with pytest.raises(ValueError):
        resolve_particle_mass(config, particle_count)
