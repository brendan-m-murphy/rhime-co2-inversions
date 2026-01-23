# RHIME CO2 Inversions 

Regional Hierarchical Inverse Modelling Environment for CO2 flux estimation.

## Installation

### Development Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/brendan-m-murphy/rhime-co2-inversions.git
cd rhime-co2-inversions
pip install -e .
```

### Running Examples

After installation, you can run examples:

```bash
python examples/example_run_rhime_202101.py
```

Or import in your own scripts:

```python
from rhime_inversions import rhime_co2

# Use the package...
```

## Dependencies

This package requires:
- Python >= 3.9
- NumPy, Pandas, Xarray, Dask
- PyMC >= 5.0, ArviZ, PyTensor
- SciPy
- OpenGHG

### Optional: OpenGHG Inversions

Some functionality (currently commented out in the code) requires OpenGHG Inversions, which is not available on PyPI. If needed, install it separately:

```bash
git clone https://github.com/openghg/openghg_inversions.git
pip install -e openghg_inversions
```

