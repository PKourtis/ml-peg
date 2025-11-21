"""Run calculation for high pressure hydrogen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import read
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.md import NPT
from janus_core.calculations.single_point import SinglePoint
import numpy as np
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

DENS_FACT = (units.m / 1.0e2) ** 3 / units.mol
PRES_RAMP = np.linspace(8, 452.5, 128)


@pytest.mark.parametrize("mlip", MODELS.items())
def relax_struct(struct: Atoms, mlip: tuple[str, Any], pressure: float) -> Atoms:
    """
    Run geometry optimisation.

    Parameters
    ----------
    struct
        Initial molecular hydrogen crystal structure.
    mlip
        Name of model use and model to get calculator.
    pressure
        The pressure the initial crystal will be optimised.

    Returns
    -------
    Atoms
        Built bulk crystal.
    """
    model_name, model = mlip
    arch = model.class_name
    calc = model.get_calculator()

    geom_opt = GeomOpt(
        struct=struct,
        arch=arch,  # will this work???
        model_path=calc,
        filter_kwargs={"scalar_pressure": pressure},
    )
    geom_opt.run()
    return geom_opt.struct


def create_triangular_cell(struct: Atoms) -> None:
    """
    Convert cell to be upper triangular.

    Based on https://github.com/CederGroupHub/chgnet/blob/main/chgnet/model/dynamics.py.

    Parameters
    ----------
    struct
        Optimised molecular hydrogen crystal structure.
    """
    a, b, c, alpha, beta, gamma = struct.cell.cellpar()
    angles = np.radians((alpha, beta, gamma))
    sin_a, sin_b, _sin_g = np.sin(angles)
    cos_a, cos_b, cos_g = np.cos(angles)
    cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
    cos_p = np.clip(cos_p, -1, 1)
    sin_p = (1 - cos_p**2) ** 0.5
    new_basis = [
        (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
        (0, b * sin_a, b * cos_a),
        (0, 0, c),
    ]
    struct.set_cell(new_basis, scale_atoms=True)
    struct.wrap()


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_dynamic_md(mlip: tuple[str, Any]) -> None:
    """
    Run calculations required for dynamic high pressure hydrogen tests.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    struct_path = DATA_PATH / "Hstart_small.xyz"

    # Relax
    pressure = PRES_RAMP[0]
    struct = relax_struct(struct_path, mlip, pressure)

    # Transform for NPT
    create_triangular_cell(struct)

    # Equilibrate
    equil = NPT(
        struct=struct,
        temp_start=50,
        temp_end=300,
        temp_step=25,
        temp_time=100,
        steps=500,
        timestep=0.2,
        stats_every=100,
        traj_every=100,
        thermostat_time=50,
        bulk_modulus=10,
        barostat_time=1500,
        pressure=pressure,
        file_prefix=OUT_PATH / f"H-MD-{mlip}",
    )
    equil.run()
    steps = equil.dyn.nsteps

    for pressure in PRES_RAMP[1:]:
        steps += 15000 // 100
        npt = NPT(
            struct=struct,
            steps=steps,
            timestep=0.2,
            stats_every=50,
            traj_every=50,
            traj_append=True,
            thermostat_time=50,
            bulk_modulus=10,
            barostat_time=1500,
            pressure=pressure,
            file_prefix=OUT_PATH / f"H-MD-{mlip}",
            restart=True,
            restart_auto=False,
        )
        npt.run()


@pytest.mark.parametrize("mlip", MODELS.items())
def test_static_md(mlip: tuple[str, Any]) -> None:
    """
    Run calculations required for static high pressure hydrogen tests.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    struct_path = DATA_PATH / "Hpres.xyz"

    model_name, model = mlip
    arch = model.class_name
    calc = model.get_calculator()

    structs = read(struct_path, index="::50")
    for struct in structs:
        struct.info["density"] = (
            np.sum(struct.get_masses()) / struct.get_volume() * DENS_FACT
        )

    SinglePoint(
        struct=structs,
        arch=arch,
        model_path=calc,
        write_results=True,
        file_prefix=OUT_PATH / f"H-static-{mlip}",
    ).run()
