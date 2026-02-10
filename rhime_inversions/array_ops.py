from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray  # assumed installed

from openghg_inversions.array_obs import get_xr_dummies


def make_key_decoder_dataset(
    keys: list[Hashable],
    *,
    key_dim: str,
    index_dim: str | None = None,
    decoder_var: str | None = None,
) -> xr.Dataset:
    """Create a decoder table mapping integer codes -> original key labels.

    This is meant to be merged into a gathered Dataset when key_as="indicator".

    Args:
      keys: Ordered list of original keys. Code i corresponds to keys[i].
      key_dim: Logical name of the key coordinate (e.g. "site").
      index_dim: Dimension name for the decoder table. Defaults to f"{key_dim}_index".
      decoder_var: Variable name containing labels. Defaults to f"{key_dim}_labels".

    Returns:
      Dataset with:
        - coord {index_dim} = [0..nkeys-1] (int32)
        - data var {decoder_var}({index_dim}) = labels (object)
    """
    if index_dim is None:
        index_dim = f"{key_dim}_index"
    if decoder_var is None:
        decoder_var = f"{key_dim}_labels"

    codes = np.arange(len(keys), dtype="int32")
    labels = np.asarray(keys, dtype=object)
    return xr.Dataset(
        data_vars={decoder_var: (index_dim, labels)},
        coords={index_dim: (index_dim, codes)},
    )


def attach_multiindex(
    obj: xr.DataArray | xr.Dataset,
    *,
    stack_dim: str,
    key_dim: str,
    ragged_coord_name: str,
) -> xr.DataArray | xr.Dataset:
    """Attach a pandas.MultiIndex on stack_dim using existing 1D coords.

    Args:
      obj: Object with 1D coords key_dim and ragged_coord_name on stack_dim.
      stack_dim: Name of stacked dimension (e.g. "nmeas").
      key_dim: Name of key coordinate (e.g. "site").
      ragged_coord_name: Name of ragged coordinate (e.g. "time").

    Returns:
      Same type as input, with MultiIndex on stack_dim.
    """
    midx = pd.MultiIndex.from_arrays(
        [obj[key_dim].values, obj[ragged_coord_name].values],
        names=(key_dim, ragged_coord_name),
    )
    mi_coords = xr.Coordinates.from_pandas_multiindex(midx, dim=stack_dim)
    return obj.assign_coords(mi_coords)


def encode_cf_compress(ds: xr.Dataset, *, stack_dim: str) -> xr.Dataset:
    """Encode a MultiIndex-ed stack_dim using CF compression-by-gathering.

    Args:
      ds: Dataset with a pandas.MultiIndex on stack_dim.
      stack_dim: Dimension holding the MultiIndex (e.g. "nmeas").

    Returns:
      CF-encoded Dataset via cf_xarray.encode_multi_index_as_compress.

    Raises:
      ValueError: If stack_dim is not MultiIndex-ed.
    """
    if stack_dim not in ds.indexes or not isinstance(ds.indexes[stack_dim], pd.MultiIndex):
        raise ValueError(f"{stack_dim!r} must be a pandas.MultiIndex before CF encoding.")
    return cf_xarray.encode_multi_index_as_compress(ds, idxnames=[stack_dim])


def gather_concat_dataarrays(
    da_dict: dict[Hashable, xr.DataArray],
    *,
    key_dim: str = "site",
    ragged_dim: str = "time",
    stack_dim: str = "nmeas",
    ragged_coord_name: str | None = None,
    key_as: Literal["labels", "indicator"] = "labels",
    make_multiindex: bool = False,
) -> xr.DataArray:
    """Concatenate ragged DataArrays without align/pad, by stacking along stack_dim.

    This avoids creating a dense (key_dim x ragged_dim) array. Each input is reshaped by:
      1) rename ragged_dim -> stack_dim
      2) attach 1D coords on stack_dim: key_dim and ragged_coord_name
      3) concat by position with join="override" (no coordinate union)

    Args:
      da_dict: Mapping key -> DataArray.
      key_dim: Name for the key coordinate on stack_dim.
      ragged_dim: Ragged dimension in each input (e.g. "time").
      stack_dim: Name of stacked output dimension (e.g. "nmeas").
      ragged_coord_name: Name for ragged coord on stack_dim (defaults to ragged_dim).
      key_as: "labels" to store original keys, or "indicator" for int codes.
      make_multiindex: If True, attach a MultiIndex on stack_dim for unstacking.

    Returns:
      DataArray with dims (stack_dim, ...) and coords key_dim(stack_dim),
      ragged_coord_name(stack_dim). If make_multiindex=True, stack_dim is MultiIndex-ed.
    """
    if ragged_coord_name is None:
        ragged_coord_name = ragged_dim

    keys = list(da_dict.keys())
    if not keys:
        raise ValueError("da_dict is empty")

    key_to_code = None
    if key_as == "indicator":
        key_to_code = {k: i for i, k in enumerate(keys)}

    pieces: list[xr.DataArray] = []
    key_vals: list[np.ndarray] = []
    ragged_vals: list[np.ndarray] = []

    for k, da in da_dict.items():
        if ragged_dim not in da.dims:
            raise ValueError(f"DataArray for key={k!r} has no dim {ragged_dim!r}")

        piece = da.rename({ragged_dim: stack_dim})
        n = piece.sizes[stack_dim]

        if key_as == "labels":
            key_vals.append(np.repeat(k, n))
        else:
            key_vals.append(np.repeat(key_to_code[k], n).astype("int32"))  # type: ignore[index]

        if ragged_dim in da.coords:
            ragged_vals.append(np.asarray(da[ragged_dim].values))
        else:
            ragged_vals.append(np.arange(n))

        pieces.append(piece)

    out = xr.concat(
        pieces,
        dim=stack_dim,
        join="override",
        coords="minimal",
        compat="override",
    ).assign_coords(
        {
            key_dim: (stack_dim, np.concatenate(key_vals)),
            ragged_coord_name: (stack_dim, np.concatenate(ragged_vals)),
        }
    )

    if make_multiindex:
        out = attach_multiindex(
            out, stack_dim=stack_dim, key_dim=key_dim, ragged_coord_name=ragged_coord_name
        )

    return out


def gather_concat_datasets(
    ds_dict: dict[str, xr.Dataset],
    *,
    key_dim: str = "site",
    ragged_dim: str = "time",
    stack_dim: str = "nmeas",
    ragged_coord_name: str | None = None,
    key_as: Literal["labels", "indicator"] = "labels",
    add_decoder: bool = False,
    make_multiindex: bool = False,
) -> xr.Dataset:
    """Concatenate ragged Datasets variable-by-variable without align/pad.

    Builds shared gathering coords once (key_dim and ragged_coord_name on stack_dim),
    then concatenates each data variable using gather_concat_dataarrays.

    If key_as="indicator" and add_decoder=True, a decoder table is merged in
    as a Dataset data variable (see make_key_decoder_dataset).

    Args:
      ds_dict: Mapping key -> Dataset.
      key_dim: Name for the key coordinate on stack_dim.
      ragged_dim: Ragged dimension (e.g. "time").
      stack_dim: Name of stacked output dimension (e.g. "nmeas").
      ragged_coord_name: Name for ragged coord on stack_dim (defaults to ragged_dim).
      key_as: "labels" or "indicator" for key_dim values along stack_dim.
      add_decoder: If True and key_as="indicator", merge in a decoder table Dataset.
      make_multiindex: If True, attach a MultiIndex on stack_dim.

    Returns:
      Dataset with stacked data variables and shared gathering coords.
    """
    if ragged_coord_name is None:
        ragged_coord_name = ragged_dim

    keys = list(ds_dict.keys())
    if not keys:
        raise ValueError("ds_dict is empty")

    first = next(iter(ds_dict.values()))
    if not first.data_vars:
        raise ValueError("First dataset has no data_vars")

    # Build shared 1D coords once using a probe variable.
    probe_var = next(iter(first.data_vars))
    key_to_code = None
    if key_as == "indicator":
        key_to_code = {k: i for i, k in enumerate(keys)}

    key_vals: list[np.ndarray] = []
    ragged_vals: list[np.ndarray] = []

    for k, ds in ds_dict.items():
        da = ds[probe_var]
        if ragged_dim not in da.dims:
            raise ValueError(f"Dataset for key={k!r} var={probe_var!r} has no dim {ragged_dim!r}")
        n = da.sizes[ragged_dim]

        if key_as == "labels":
            key_vals.append(np.repeat(k, n))
        else:
            key_vals.append(np.repeat(key_to_code[k], n).astype("int32"))  # type: ignore[index]

        if ragged_dim in da.coords:
            ragged_vals.append(np.asarray(da[ragged_dim].values))
        else:
            ragged_vals.append(np.arange(n))

    key_1d = np.concatenate(key_vals)
    ragged_1d = np.concatenate(ragged_vals)

    out_vars: dict[str, xr.DataArray] = {}
    for vname in first.data_vars:
        out_vars[str(vname)] = gather_concat_dataarrays(
            {k: ds[vname] for k, ds in ds_dict.items()},
            key_dim=key_dim,
            ragged_dim=ragged_dim,
            stack_dim=stack_dim,
            ragged_coord_name=ragged_coord_name,
            key_as="labels",  # temporary; we overwrite with shared coords below
            make_multiindex=False,
        )

    out = xr.Dataset(out_vars).assign_coords(
        {
            key_dim: (stack_dim, key_1d),
            ragged_coord_name: (stack_dim, ragged_1d),
        }
    )

    if key_as == "indicator" and add_decoder:
        out = xr.merge([out, make_key_decoder_dataset(keys, key_dim=key_dim)])

    if make_multiindex:
        out = attach_multiindex(
            out, stack_dim=stack_dim, key_dim=key_dim, ragged_coord_name=ragged_coord_name
        )

    return out


def get_ragged_xr_dummies(
    da: dict[str, xr.DataArray],
    categories: Sequence[Any] | pd.Index | xr.DataArray | np.ndarray | None = None,
    cat_dim: str = "categories",
    return_sparse: bool = True,
    key_dim: str = "source",
    stack_dim: str | None = None,
) -> xr.DataArray:
    """Make dummy matrices for dict of DataArrays and then gather along key_dim."""
    da_transformed = {
        k: get_xr_dummies(v, cat_dim=cat_dim, categories=categories, return_sparse=return_sparse)
        for k, v in da.items()
    }
    stack_dim = stack_dim or (key_dim + "_" + cat_dim)
    result = gather_concat_dataarrays(
        da_transformed, key_dim=key_dim, ragged_dim=cat_dim, stack_dim=stack_dim
    )
    return result
