from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import pytest
import xarray as xr

from rhime_inversions import rhime_co2
from rhime_inversions import calculate_basis_functions as cbf

from helpers import load_merged_data

here = Path(__file__).resolve().parent
data_dir = here / "big_data"
frozen_dir = data_dir / "frozen"


def get_data_dict():
    return load_merged_data(str(data_dir / "mhd_mf_obs_sim_test.nc"))


def get_basis():
    return xr.open_dataset(data_dir / "weighted_co2-code_test__EUROPE_201401.nc", engine="h5netcdf")


def get_bc_basis():
    return xr.open_dataset(data_dir / "NESW_EUROPE_2014.nc", engine="h5netcdf")


def _assert_ds_close(ds_new: xr.Dataset, ds_frozen: xr.Dataset):
    """
    Robust-ish comparison:
    - exact on dims/coords names
    - numeric variables close (tolerances)
    - non-numeric variables equal (or skipped)
    """
    # Ensure same variables (order-insensitive)
    assert set(ds_new.data_vars) == set(ds_frozen.data_vars)
    assert set(ds_new.coords) == set(ds_frozen.coords)

    # Compare coordinates exactly where possible
    for c in ds_new.coords:
        # Many coords are numeric times/ints; treat as exact for now
        xr.testing.assert_identical(ds_new[c], ds_frozen[c])

    # Compare variables
    for v in ds_new.data_vars:
        a = ds_new[v]
        b = ds_frozen[v]

        assert a.dims == b.dims
        assert a.shape == b.shape

        if np.issubdtype(a.dtype, np.number):
            xr.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
        else:
            # For strings/objects/attrs-heavy vars you might want looser checks.
            xr.testing.assert_identical(a, b)


@pytest.fixture()
def rhime_test_config():
    # NOTE: obs_dict, fp_dict, flux_dict, bc_dict are probably not needed because we are mocking get_mf_obs_sims
    obs_dict = {
        "species": "co2",
        "site": ["MHD"],
        "inlet": ["24m"],
        "averaging_period": ["4h"],
        "instrument": ["multiple"],
        "data_level": ["2"],
        "store": "obs_nir_2024_01_25_store_zarr",
        "calibration_scale": None,
        "start_date": "2014-01-01",
        "end_date": "2014-01-02",
        "filters": ["daytime"],
    }

    fp_dict = {
        "species": "co2",
        "domain": "EUROPE",
        "site": ["MHD"],
        "fp_height": ["10m"],
        "start_date": "2014-01-01",
        "end_date": "2014-01-02",
        "store": "uk_co2_footprints_202406",
    }

    flux_dict = {
        "species": "co2",
        "domain": "EUROPE",
        "source": ["edgar-fossil-hrly-flat", "vprm-gee", "vprm-resp"],
        "start_date": "2014-01-01",
        "end_date": "2014-02-01",
        "store": "uk_co2_zarr_store",
        "flux_sf": {
            "edgar-fossil-hrly-flat": 1.0,
            "vprm-gee": 1.0,
            "vprm-resp": 1.0,
        },
        "sector_dict": {
            "fossil": "edgar-fossil-hrly-flat",
            "gee": "vprm-gee",
            "resp": "vprm-resp",
        },
    }

    # Boundary conditions dictionary specifications
    bc_dict = {
        "species": "co2",
        "domain": "EUROPE",
        "bc_input": "camsv22-co2",
        "bc_freq": "monthly",
        "start_date": "2014-01-01",
        "end_date": "2014-02-01",
        "store": "uk_co2_zarr_store",
        "bc_sf": None,
    }

    # Basis functions dictionary
    basis_dict = {
        "fp_basis_case": "test",
        "basis_directory": "test",
        "fp_basis_algorithm": None,
        "nbasis": [50, 50, 50],
        "bc_basis_case": "NESW",
        "bc_basis_directory": "/group/chemistry/acrg/LPDM/bc_basis_functions/",
    }

    # MCMC dict
    mcmc_dict = {
        "xprior": {
            "edgar-fossil-hrly-flat": {"pdf": "truncatednormal", "mu": 1.2, "sigma": 1.2, "lower": 0.0},
            "vprm-gee": {"pdf": "truncatednormal", "mu": 1.0, "sigma": 2.0, "lower": 0.0},
            "vprm-resp": {"pdf": "truncatednormal", "mu": 1.0, "sigma": 2.0, "lower": 0.0},
        },
        "bcprior": {"pdf": "truncatednormal", "lower": 0.0, "mu": 1.0, "sigma": 0.05},
        "sigprior": {"pdf": "uniform", "lower": 0.1, "upper": 3.0},
        "add_offset": False,
        "offsetprior": None,
        "nit": 10,
        "burn": 1,
        "tune": 10,
        "nchain": 2,
        "sigma_per_site": True,
    }

    return obs_dict, fp_dict, flux_dict, bc_dict, basis_dict, mcmc_dict


@pytest.mark.data_required
@pytest.mark.parametrize("site", ["MHD"])
def test_basis_functions_wrapper_frozen(tmp_path, mocker, site, rhime_test_config):
    data_dict = get_data_dict()
    basis_ds = get_basis()
    bc_basis_ds = get_bc_basis()

    # Mock slow basis computation
    # Adjust return signatures if cbf.basis() returns (ds, dir) etc.
    mocker.patch("rhime_inversions.calculate_basis_functions.basis", return_value=basis_ds)
    mocker.patch(
        "rhime_inversions.calculate_basis_functions.basis_boundary_conditions", return_value=bc_basis_ds
    )

    # --- minimal-ish inputs; adapt to your wrapper signature ---
    # If wrapper needs obs_dict/flux_dict/fp_dict/bc_dict/basis_dict, pass your test ones.
    # I'm showing a generic pattern; replace with your real args.
    obs_dict, fp_dict, flux_dict, _, basis_dict, _ = rhime_test_config

    # fixes performed in rhime_co2.rhime_inversions
    basis_dict["site"] = obs_dict["site"]
    basis_dict["source"] = flux_dict["source"]
    basis_dict["domain"] = fp_dict["domain"]
    basis_dict["start_date"] = obs_dict["start_date"]

    t0 = time.perf_counter()
    fp_data, tempdir, basis_dir, bc_basis_dir = cbf.basis_functions_wrapper(
        data_dict=data_dict,
        basis_dict=basis_dict,
        use_bc=True,
        outputpath=str(tmp_path),
        outputname="TEST_",
    )
    runtime = time.perf_counter() - t0

    # basic sanity checks
    assert site in fp_data
    assert isinstance(fp_data[site], xr.Dataset)

    # compare to frozen
    frozen_path = frozen_dir / f"fp_data_{site}.nc"
    frozen = xr.open_dataset(frozen_path, engine="h5netcdf")

    # might want to be more lax...
    # vars_to_check = ["H", "H_bc", "mf", "mf_mod"]  # adjust
    # _assert_ds_close(fp_data[site][vars_to_check], frozen[vars_to_check])

    _assert_ds_close(fp_data[site], frozen)

    # record runtime as an informational assertion (tune threshold to your machine/CI)
    # Better: use pytest-benchmark below.
    assert runtime < 5.0, f"basis_functions_wrapper too slow: {runtime:.2f}s"


@pytest.mark.data_required
def test_example_config_post_data(tmpdir, mocker, rhime_test_config):
    data_dict = get_data_dict()
    basis_ds = get_basis()
    bc_basis_ds = get_bc_basis()

    # 1) mock the obs/sim gather step
    mocker.patch(
        "rhime_inversions.rhime_co2.get_mf_obs_sims",
        return_value=(
            data_dict,
            ["MHD"],  # sites
            ["24m"],  # inlets
            ["10m"],  # fp_heights
            ["multiple"],  # instruments
            ["4h"],  # averaging periods
        ),
    )

    # 2) mock basis() and basis_boundary_conditions()
    #
    # IMPORTANT: adjust return_value to match the real function’s signature.
    # Common patterns are:
    #   basis(...) -> basis_ds
    #   basis(...) -> (basis_ds, basis_dir)
    #   basis_boundary_conditions(...) -> bc_basis_ds
    #   basis_boundary_conditions(...) -> (bc_basis_ds, bc_basis_dir)
    mocker.patch(
        "rhime_inversions.rhime_co2.cbf.basis",
        return_value=basis_ds,  # or (basis_ds, "ignored/path")
    )
    mocker.patch(
        "rhime_inversions.rhime_co2.cbf.basis_boundary_conditions",
        return_value=bc_basis_ds,  # or (bc_basis_ds, "ignored/path")
    )

    # OPTIONAL: if you also want to avoid running the actual inversion/MCMC
    mocker.patch("rhime_inversions.rhime_co2.inferpymc", return_value={"dummy": True})
    mocker.patch("rhime_inversions.rhime_co2.inferpymc_postprocessouts", return_value=None)

    obs_dict, fp_dict, flux_dict, bc_dict, basis_dict, mcmc_dict = rhime_test_config

    use_bc = True
    model_error_method = "residual"
    sigma_freq = None

    outputname = "TEST_"
    outputpath = str(tmpdir)
    country_file = str(data_dir / "country_EUROPE_EEZ_PARIS_gapfilled.nc")

    # recompute basis functions...
    basis_dict["fp_basis_algorithm"] = "weighted"
    basis_dict["fp_basis_case"] = None

    rhime_co2.rhime_inversions(
        obs_dict=obs_dict,
        flux_dict=flux_dict,
        bc_dict=bc_dict,
        fp_dict=fp_dict,
        basis_dict=basis_dict,
        mcmc_dict=mcmc_dict,
        use_bc=use_bc,
        model_error_method=model_error_method,
        sigma_freq=sigma_freq,
        outputname=outputname,
        outputpath=outputpath,
        country_file=country_file,
    )


@pytest.mark.data_required
@pytest.mark.slow
@pytest.mark.parametrize(
    "sector_key, nbasis",
    [
        ("edgar-fossil-hrly-flat", 50),
        ("vprm-gee", 50),
        ("vprm-resp", 50),
    ],
)
def test_generate_weighted_basis_per_sector_timed(tmp_path, sector_key, nbasis):
    """
    Exercise the slow path: create basis functions (weighted/bucket basis)
    with fp_basis_case=None and fp_basis_algorithm='weighted'.

    We call the underlying algorithm per-sector so we can time each sector.
    """
    data_dict = get_data_dict()

    # minimal metadata needed by bucketbasisfunction
    site = ["MHD"]
    domain = "EUROPE"
    start_date = "2014-01-01"
    outputname = f"TEST_BASIS_{sector_key.replace('-', '_')}_"

    # Run only one sector at a time, so timing is per-sector
    sources = [sector_key]
    nbasis_one = [nbasis]

    t0 = time.perf_counter()
    outdir = cbf.bucketbasisfunction(
        emissions_name=sources,
        data_dict=data_dict,
        sites=site,
        start_date=start_date,
        domain=domain,
        species="CO2",
        outputname=outputname,
        outputdir=str(tmp_path),  # put outputs in the test tmpdir
        nbasis=nbasis_one,
    )
    dt = time.perf_counter() - t0

    # Timing info (shows up with -s, or in CI logs if captured)
    print(f"[timing] sector={sector_key} nbasis={nbasis} dt={dt:.2f}s outdir={outdir}")

    # The algorithm writes a file; check it exists and is readable
    # Filenames from code:
    # weighted_co2-{outputname}_{domain}_{YYYY}{MM}.nc
    result_file = tmp_path / domain / f"weighted_co2-{outputname}_{domain}_201401.nc"
    assert result_file.exists(), f"Expected basis file not written: {result_file}"

    ds = xr.open_dataset(result_file, engine="h5netcdf")

    # Basic invariants
    assert "basis" in ds.data_vars
    assert set(ds["basis"].dims) == {"sector", "lat", "lon", "time"}
    assert ds.sizes["sector"] == 1
    assert ds.sizes["time"] == 1

    # basis values should be positive integers (0 may appear depending on algorithm; allow >=0)
    b = ds["basis"].values
    assert np.isfinite(b).all()
    assert b.min() >= 0
    assert b.max() > 1  # should have multiple regions if nbasis ~50

    # Optional: keep artifacts locally for debugging
    if os.environ.get("RHIME_KEEP_BASIS_ARTIFACTS", ""):
        keep_dir = Path(os.environ["RHIME_KEEP_BASIS_ARTIFACTS"]).expanduser()
        keep_dir.mkdir(parents=True, exist_ok=True)
        result_file.rename(keep_dir / result_file.name)
