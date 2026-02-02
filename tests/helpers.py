import xarray as xr
from openghg.dataobjects import FluxData, BoundaryConditionsData


list_keys = [
    "site",
    "inlet",
    "instrument",
    "sampling_period",
    "sampling_period_unit",
    "averaged_period_str",
    "scale",
    "network",
    "data_owner",
    "data_owner_email",
]


def fp_all_from_dataset(ds: xr.Dataset) -> dict:
    """Recover "fp_all" dictionary from "combined scenario" dataset.

    This is the inverse of `make_combined_scenario`, except that the attributes of the
    scenarios, fluxes, and boundary conditions may be different.

    Args:
        ds: dataset created by `make_combined_scenario`

    Returns:
        dictionary containing model scenarios keyed by site, as well as flux and boundary conditions.
    """
    fp_all = {}

    # we'll get scales as we get scenarios
    fp_all[".scales"] = {}

    # get scenarios
    bc_vars = ["vmr_n", "vmr_e", "vmr_s", "vmr_w"]

    for i, site in enumerate(ds.site.values):
        scenario = (
            ds.sel(site=site, drop=True).drop_vars(["flux", *bc_vars], errors="ignore").drop_dims("source")
        )

        # extract attributes that were gathered into a list
        for k in list_keys:
            try:
                val = scenario.attrs[k][i]
            except (ValueError, IndexError):
                val = "None"

            if k == "scale":
                fp_all[".scales"][site] = val
            else:
                scenario.attrs[k] = val

        fp_all[str(site)] = scenario.dropna("time", subset=["mf"])

    # get fluxes
    fp_all[".flux"] = {}

    for i, source in enumerate(ds.source.values):
        try:
            flux_ds = (
                ds[["flux"]]  # double brackets to get dataset
                .sel(source=source, drop=True)
                .expand_dims({"time": [ds.time.min().values]})
                .transpose(..., "time")
            )
        except:
            flux_ds = ds[["flux"]].sel(source=source, drop=True)

        # extract attributes that were gathered into a list
        for k in list_keys:
            try:
                val = flux_ds.attrs[k][i]
            except (ValueError, IndexError):
                val = "None"
            flux_ds.attrs[k] = val

        fp_all[".flux"][str(source)] = FluxData(data=flux_ds, metadata={"data_type": "flux"})

    try:
        bc_ds = ds[bc_vars]
    except KeyError:
        pass
    else:
        if "time" not in bc_ds.dims:
            bc_ds = bc_ds.expand_dims({"time": [ds.time.min().values]})

        fp_all[".bc"] = BoundaryConditionsData(data=bc_ds, metadata={})

    species = ds.attrs.get("species", None)
    if species is not None:
        species = species.upper()
    fp_all[".species"] = species

    try:
        fp_all[".units"] = float(ds.mf.attrs.get("units", 1.0))
    except ValueError:
        # conversion to float failed
        fp_all[".units"] = 1.0

    return fp_all


def load_merged_data(data_path: str) -> dict:
    return fp_all_from_dataset(xr.open_dataset(data_path, engine="h5netcdf"))
