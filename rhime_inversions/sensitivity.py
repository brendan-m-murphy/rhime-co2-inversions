from __future__ import annotations

import numpy as np
import xarray as xr

from openghg_inversions.basis._helpers import bc_sensitivity as oi_bc_sensitivity
from openghg_inversions.basis._helpers import fp_sensitivity as oi_fp_sensitivity

from .logging_utils import log_step, setup_rhime_logger


logger = setup_rhime_logger(__name__)


def _normalise_basis_array(basis_func: xr.DataArray, flux_sources: list[str]) -> xr.DataArray:
    basis_func = basis_func.rename("basis")

    if "sector" in basis_func.dims and "source" not in basis_func.dims:
        basis_func = basis_func.rename(sector="source")

    if "source" in basis_func.dims and basis_func.sizes["source"] == len(flux_sources):
        basis_func = basis_func.assign_coords(source=flux_sources)

    return basis_func


def _get_basis_for_source(basis_func: xr.DataArray, source: str) -> xr.DataArray:
    if "source" not in basis_func.dims:
        return basis_func

    if source in basis_func.coords["source"].values:
        return basis_func.sel(source=source, drop=True)

    return basis_func.isel(source=0, drop=True)


def _basis_region_value(value):
    if isinstance(value, (np.integer, int)):
        return int(value) + 1
    return value.decode("ascii") if isinstance(value, bytes) else value


def _region_label(value) -> str:
    return str(_basis_region_value(value))


def _prefixed_region_names(region_values: np.ndarray, source: str | None) -> list[str]:
    labels: list[str] = []
    for value in np.asarray(region_values):
        label = _region_label(value)
        labels.append(f"{source}-{label}" if source else label)
    return labels


def _compute_herr(fp_x_flux: xr.DataArray, basis_func: xr.DataArray, region_values: np.ndarray) -> xr.DataArray:
    if "time" in basis_func.dims and basis_func.sizes.get("time", 0) == 1:
        basis_func = basis_func.squeeze("time", drop=True)

    ntime = fp_x_flux.sizes["time"]

    if "region" in basis_func.dims:
        base_v = basis_func.values.reshape(len(fp_x_flux.lat) * len(fp_x_flux.lon), len(basis_func.region))
        h_all_v = fp_x_flux.values.reshape(len(fp_x_flux.lat) * len(fp_x_flux.lon), ntime)
        herr = np.zeros((len(basis_func.region), ntime))
        region_iter = range(len(basis_func.region))

        for i in region_iter:
            region_contrib = h_all_v * base_v[:, i, np.newaxis]
            with np.errstate(divide="ignore", invalid="ignore"):
                region_logs = np.log(np.abs(region_contrib))
            region_logs[~np.isfinite(region_logs)] = np.nan
            s_ln = np.nanstd(region_logs, axis=0)
            coeff_var_ln = np.sqrt(np.exp(s_ln**2) - 1)
            herr[i, :] = np.abs(np.nan_to_num(coeff_var_ln))
    else:
        labels = basis_func.values.reshape(len(fp_x_flux.lat) * len(fp_x_flux.lon))
        h_all_v = fp_x_flux.values.reshape(len(fp_x_flux.lat) * len(fp_x_flux.lon), ntime)
        herr = np.zeros((len(region_values), ntime))

        for i, region in enumerate(np.asarray(region_values)):
            region_label = _basis_region_value(region)
            region_idx = np.where(labels == region_label)[0]
            if region_idx.size == 0:
                continue
            region_contrib = h_all_v[region_idx, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                region_logs = np.log(np.abs(region_contrib))
            region_logs[~np.isfinite(region_logs)] = np.nan
            s_ln = np.nanstd(region_logs, axis=0)
            coeff_var_ln = np.sqrt(np.exp(s_ln**2) - 1)
            herr[i, :] = np.abs(np.nan_to_num(coeff_var_ln))

    return xr.DataArray(
        herr,
        dims=("region", "time"),
        coords={"region": region_values, "time": fp_x_flux.time.values},
    )


def fp_sensitivity(
    data_dict: dict,
    domain: str,
    basis_case: str | None,
    basis_directory=None,
    basis_func: xr.DataArray | None = None,
    verbose=True,
) -> dict:
    sites = [key for key in list(data_dict.keys()) if not key.startswith(".")]
    flux_sources = list(data_dict[".flux"].keys())

    if basis_func is None:
        from . import calculate_basis_functions as cbf

        with log_step(logger, f"fp_sensitivity: read basis functions (domain={domain} basis_case={basis_case})"):
            basis_func = cbf.basis(
                domain=domain,
                basis_case=basis_case,
                basis_directory=basis_directory,
            ).basis

    basis_func = _normalise_basis_array(basis_func, flux_sources)

    with log_step(logger, "fp_sensitivity: apply upstream basis helper"):
        data_dict = oi_fp_sensitivity(data_dict, basis_func=basis_func)

    for site in sites:
        site_ds = data_dict[site]
        sensitivity = site_ds["H"]

        if "H" in site_ds:
            site_ds = site_ds.drop_vars("H")
        if "Herr" in site_ds:
            site_ds = site_ds.drop_vars("Herr")
        if "region" in site_ds.dims:
            site_ds = site_ds.drop_dims("region")
        data_dict[site] = site_ds

        if "source" in sensitivity.dims:
            sensitivity = sensitivity.transpose("source", "region", "time")
            fp_var = site_ds["fp_x_flux_sectoral"]
            h_parts: list[xr.DataArray] = []
            herr_parts: list[xr.DataArray] = []

            for source in flux_sources:
                source_h = sensitivity.sel(source=source, drop=True)
                source_basis = _get_basis_for_source(basis_func, source)
                source_herr = _compute_herr(
                    fp_var.sel(source=source, drop=True),
                    source_basis,
                    source_h.region.values,
                )
                region_names = _prefixed_region_names(source_h.region.values, source)
                h_parts.append(source_h.assign_coords(region=region_names))
                herr_parts.append(source_herr.assign_coords(region=region_names))

            data_dict[site]["H"] = xr.concat(h_parts, dim="region")
            data_dict[site]["Herr"] = xr.concat(herr_parts, dim="region")
        else:
            source = flux_sources[0] if flux_sources else None
            fp_var_name = "fp_x_flux" if "fp_x_flux" in site_ds else f"Hall_{source}"
            source_herr = _compute_herr(site_ds[fp_var_name], basis_func, sensitivity.region.values)
            region_names = _prefixed_region_names(sensitivity.region.values, source)
            data_dict[site]["H"] = sensitivity.assign_coords(region=region_names)
            data_dict[site]["Herr"] = source_herr.assign_coords(region=region_names)

    data_dict[".basis"] = basis_func
    return data_dict


def bc_sensitivity(
    data_dict,
    domain,
    basis_case,
    bc_basis_directory=None,
):
    with log_step(logger, f"bc_sensitivity: apply upstream helper (basis_case={basis_case})"):
        return oi_bc_sensitivity(
            data_dict,
            domain=domain,
            basis_case=basis_case,
            bc_basis_directory=bc_basis_directory,
        )
