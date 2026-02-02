from pathlib import Path
import xarray as xr

from rhime_inversions import calculate_basis_functions as cbf
from helpers import load_merged_data

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "big_data"
FROZEN_DIR = DATA_DIR / "frozen"
FROZEN_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data_dict = load_merged_data(str(DATA_DIR / "mhd_mf_obs_sim_test.nc"))
    basis_ds = xr.open_dataset(DATA_DIR / "weighted_co2-code_test__EUROPE_201401.nc", engine="h5netcdf")
    bc_basis_ds = xr.open_dataset(DATA_DIR / "NESW_EUROPE_2014.nc", engine="h5netcdf")

    # DO NOT MOCK here; instead pass the datasets by temporarily monkeypatching
    # or (better) refactor wrapper to accept basis datasets optionally.
    #
    # For now: quick-and-dirty: call fp_sensitivity/bc_sensitivity directly if they accept basis ds.

    # If you *must* use mocking outside pytest, simplest is: temporarily replace functions.
    old_basis = cbf.basis
    old_bc_basis = cbf.basis_boundary_conditions
    cbf.basis = lambda *args, **kwargs: basis_ds
    cbf.basis_boundary_conditions = lambda *args, **kwargs: bc_basis_ds
    try:
        basis_dict = {
            "fp_basis_case": "test",
            "basis_directory": "test",
            "fp_basis_algorithm": None,
            "nbasis": [50, 50, 50],
            "bc_basis_case": "NESW",
            "bc_basis_directory": "test",
            "domain": "EUROPE",
            "source": ["edgar-fossil-hrly-flat", "vprm-gee", "vprm-resp"],
            "site": ["MHD"],
            "start_date": "2014-01-01",
        }

        fp_data, *_ = cbf.basis_functions_wrapper(
            data_dict=data_dict,
            basis_dict=basis_dict,
            use_bc=True,
            outputpath=str(HERE / "tmp_out"),
            outputname="FROZEN_",
        )
        fp_data["MHD"].to_netcdf(FROZEN_DIR / "fp_data_MHD.nc", engine="h5netcdf")
        print("Wrote", FROZEN_DIR / "fp_data_MHD.nc")
    finally:
        cbf.basis = old_basis
        cbf.basis_boundary_conditions = old_bc_basis


if __name__ == "__main__":
    main()
