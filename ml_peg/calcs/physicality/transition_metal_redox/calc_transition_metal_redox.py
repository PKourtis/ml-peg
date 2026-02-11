"""Run calculation for transition_metal_redox."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read
from aseMolec import anaAtoms
from janus_core.calculations.md import NPT
import numpy as np
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

TRANSITION_METALS = ["Co"]#, "Cr", "Fe", "Mn", "V", "Ti"]


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_transition_metal_water_md(mlip: tuple[str, Any]) -> None:
    """
    Run MLMD of aqueous transition metals in multiple oxidation and spin states.

    Parameters
    ----------
    mlip
        Name of model use and model.
    """

    model_name, model = mlip
    model.device = "cuda"
    model.default_dtype = "float32"
    model.kwargs["enable_cueq"] = True
    # model.kwargs["charges_key"]
    # model.kwargs["spin_key"]

    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    for tmetal in TRANSITION_METALS:
        struct_path = DATA_PATH / f"{salt}_start.xyz"
        struct = read(struct_path, "0")
        struct.calc = calc

        npt = NPT(
            struct=struct,
            steps=40000,
            timestep=0.5,
            stats_every=50,
            traj_every=100,
            traj_append=True,
            thermostat_time=50,
            barostat_time=None,
            file_prefix=OUT_PATH / f"{salt}_{model_name}",
        )
        npt.run()