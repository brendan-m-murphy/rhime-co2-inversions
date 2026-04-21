from __future__ import annotations

import os
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from openghg_inversions.basis._functions import basis as oi_basis
from openghg_inversions.basis._functions import basis_boundary_conditions as oi_basis_boundary_conditions
from openghg_inversions.basis._functions import bucketbasisfunction as oi_bucketbasisfunction
from openghg_inversions.basis._functions import quadtreebasisfunction as oi_quadtreebasisfunction

try:  # Forward-compatible with newer openghg_inversions releases.
    from openghg_inversions.basis.basis_functions import BasisFunctions
except (ModuleNotFoundError, ImportError):  # pragma: no cover - older released dependency.
    BasisFunctions = None  # type: ignore[assignment]

from .sensitivity import bc_sensitivity, fp_sensitivity


BasisFunction = namedtuple("BasisFunction", ["description", "algorithm"])


def basis(domain: str, basis_case: str, basis_directory: Optional[str] = None) -> xr.Dataset:
    return oi_basis(domain=domain, basis_case=basis_case, basis_directory=basis_directory)


def basis_boundary_conditions(
    domain: str,
    basis_case: str,
    bc_basis_directory: Optional[str] = None,
) -> xr.Dataset:
    return oi_basis_boundary_conditions(
        domain=domain,
        basis_case=basis_case,
        bc_basis_directory=bc_basis_directory,
    )


def _normalise_nbasis(nbasis: int | list[int], nsectors: int) -> list[int]:
    if isinstance(nbasis, int):
        return [nbasis] * nsectors
    if len(nbasis) != nsectors:
        raise ValueError("`nbasis` must be an int or a list with one entry per source.")
    return [int(value) for value in nbasis]


def _default_outputdir() -> str:
    return tempfile.mkdtemp(prefix="Temp_", dir=os.getcwd())


def _basis_filename(
    basis_algorithm: str,
    species: str,
    domain: str,
    basis_data_array: xr.DataArray,
    outputname: str | None,
) -> str:
    stamp = pd.to_datetime(basis_data_array.time.min().values).strftime("%Y%m")
    prefix = f"{basis_algorithm}_{species.lower()}"
    if outputname:
        return f"{prefix}-{outputname}_{domain}_{stamp}.nc"
    return f"{prefix}_{domain}_{stamp}.nc"


def _save_basis_dataset(
    basis_data_array: xr.DataArray,
    basis_algorithm: str,
    species: str,
    domain: str,
    outputdir: str,
    outputname: str | None,
) -> None:
    output_path = Path(outputdir) / domain
    output_path.mkdir(parents=True, exist_ok=True)

    basis_to_save = basis_data_array
    if "source" in basis_to_save.dims and "sector" not in basis_to_save.dims:
        basis_to_save = basis_to_save.rename(source="sector")

    basis_to_save.to_dataset(name="basis").to_netcdf(
        output_path / _basis_filename(basis_algorithm, species, domain, basis_to_save, outputname),
        mode="w",
    )


def _normalise_basis_array(basis_data_array: xr.DataArray, source: list[str] | None) -> xr.DataArray:
    basis_data_array = basis_data_array.rename("basis")

    if "sector" in basis_data_array.dims and "source" not in basis_data_array.dims:
        basis_data_array = basis_data_array.rename(sector="source")

    if "source" in basis_data_array.dims and source is not None and basis_data_array.sizes["source"] == len(source):
        basis_data_array = basis_data_array.assign_coords(source=source)

    return basis_data_array


def _run_sectoral_basis_algorithm(
    algorithm,
    emissions_name: list[str],
    data_dict: dict,
    start_date: str,
    domain: str,
    nbasis: int | list[int],
) -> xr.DataArray:
    if len(emissions_name) == 1:
        return algorithm(
            data_dict,
            start_date,
            domain,
            emissions_name=emissions_name,
            nbasis=int(_normalise_nbasis(nbasis, 1)[0]),
        )

    basis_arrays: list[xr.DataArray] = []
    for source, nregion in zip(emissions_name, _normalise_nbasis(nbasis, len(emissions_name)), strict=True):
        sector_basis = algorithm(
            data_dict,
            start_date,
            domain,
            emissions_name=[source],
            nbasis=int(nregion),
        )
        basis_arrays.append(sector_basis.expand_dims(source=[source]))

    return xr.concat(basis_arrays, dim="source")


def _basis_case_name(basis_algorithm: str, species: str, outputname: str | None) -> str:
    prefix = f"{basis_algorithm}_{species.lower()}"
    if outputname:
        return f"{prefix}-{outputname}"
    return prefix


def _extract_flux_dataarray(flux_entry):
    if hasattr(flux_entry, "data") and isinstance(flux_entry.data, xr.Dataset) and "flux" in flux_entry.data:
        return flux_entry.data["flux"]
    if isinstance(flux_entry, xr.Dataset) and "flux" in flux_entry:
        return flux_entry["flux"]
    if isinstance(flux_entry, xr.DataArray):
        return flux_entry
    return None


def _build_basis_functions_object(data_dict: dict, basis_data_array: xr.DataArray):
    if BasisFunctions is None or not hasattr(BasisFunctions, "from_basis_flat"):
        return None

    flux_entries = data_dict.get(".flux", {})
    if not flux_entries:
        return None

    if "source" in basis_data_array.dims and hasattr(BasisFunctions, "from_multi_source_basis_flat"):
        basis_map = {}
        flux_map = {}
        for source in basis_data_array.coords["source"].values:
            flux_da = _extract_flux_dataarray(flux_entries.get(source))
            if flux_da is None:
                return None
            basis_map[str(source)] = basis_data_array.sel(source=source, drop=True).squeeze("time", drop=True)
            flux_map[str(source)] = flux_da
        return BasisFunctions.from_multi_source_basis_flat(basis_map, flux_map)

    first_source = next(iter(flux_entries))
    flux_da = _extract_flux_dataarray(flux_entries[first_source])
    if flux_da is None:
        return None

    basis_flat = basis_data_array.squeeze("time", drop=True) if "time" in basis_data_array.dims else basis_data_array
    return BasisFunctions.from_basis_flat(
        basis_flat=basis_flat,
        flux=flux_da,
        operator_kwargs={"state_dim": "region"},
    )


def quadtreebasisfunction(
    emissions_name,
    data_dict,
    sites,
    start_date,
    domain,
    species,
    outputname,
    outputdir=None,
    nbasis=50,
):
    outputdir = outputdir or _default_outputdir()
    basis_data_array = _run_sectoral_basis_algorithm(
        oi_quadtreebasisfunction,
        emissions_name=emissions_name,
        data_dict=data_dict,
        start_date=start_date,
        domain=domain,
        nbasis=nbasis,
    )
    _save_basis_dataset(basis_data_array, "quadtree", species, domain, outputdir, outputname)
    return outputdir


def bucketbasisfunction(
    emissions_name,
    data_dict,
    sites,
    start_date,
    domain,
    species,
    outputname,
    outputdir=None,
    nbasis=50,
):
    outputdir = outputdir or _default_outputdir()
    basis_data_array = _run_sectoral_basis_algorithm(
        oi_bucketbasisfunction,
        emissions_name=emissions_name,
        data_dict=data_dict,
        start_date=start_date,
        domain=domain,
        nbasis=nbasis,
    )
    _save_basis_dataset(basis_data_array, "weighted", species, domain, outputdir, outputname)
    return outputdir


basis_functions = {
    "quadtree": BasisFunction("quadtree algorithm", quadtreebasisfunction),
    "weighted": BasisFunction("weighted by data algorithm", bucketbasisfunction),
}


def basis_functions_wrapper(
    data_dict: dict,
    basis_dict: dict,
    use_bc: bool,
    outputname: Optional[str] = None,
    outputpath: Optional[str] = None,
):
    fp_basis_case = basis_dict["fp_basis_case"]
    basis_directory = basis_dict["basis_directory"]
    basis_algorithm = basis_dict["fp_basis_algorithm"]
    nbasis = basis_dict["nbasis"]
    bc_basis_case = basis_dict["bc_basis_case"]
    bc_basis_directory = basis_dict["bc_basis_directory"]
    domain = basis_dict["domain"]
    source = basis_dict["source"]
    start_date = basis_dict["start_date"]

    tempdir = None
    basis_data_array: xr.DataArray | None = None

    if fp_basis_case is not None:
        if basis_algorithm:
            print(
                f"Basis algorithm {basis_algorithm} and basis case {fp_basis_case} supplied; using {fp_basis_case}."
            )
        basis_data_array = _normalise_basis_array(
            basis(domain=domain, basis_case=fp_basis_case, basis_directory=basis_directory).basis,
            source=source,
        )
    elif basis_algorithm is None:
        raise ValueError("One of `fp_basis_case` or `basis_algorithm` must be specified.")
    else:
        try:
            basis_function = basis_functions[basis_algorithm]
        except KeyError as exc:
            raise ValueError(
                "Basis algorithm not recognised. Please use either 'quadtree' or 'weighted', or input a basis function file"
            ) from exc

        print(f"Using {basis_function.description} to derive basis functions.")
        tempdir = basis_function.algorithm(
            source,
            data_dict,
            basis_dict.get("site"),
            start_date,
            domain,
            "CO2",
            outputname,
            outputpath,
            nbasis,
        )
        basis_directory = tempdir
        fp_basis_case = _basis_case_name(basis_algorithm, "co2", outputname)
        basis_data_array = _normalise_basis_array(
            basis(domain=domain, basis_case=fp_basis_case, basis_directory=basis_directory).basis,
            source=source,
        )

    fp_data = fp_sensitivity(
        data_dict,
        domain,
        basis_case=fp_basis_case,
        basis_directory=basis_directory,
        basis_func=basis_data_array,
    )

    basis_object = _build_basis_functions_object(data_dict, basis_data_array) if basis_data_array is not None else None
    if basis_object is not None:
        fp_data[".basis_functions"] = basis_object

    if use_bc:
        fp_data = bc_sensitivity(
            fp_data,
            domain=domain,
            basis_case=bc_basis_case,
            bc_basis_directory=bc_basis_directory,
        )

    return fp_data, tempdir, basis_directory, bc_basis_directory
