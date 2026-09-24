# Running the workflow

All commands below must be run after creating the Conda environment:

```bash
conda env create -f environment.yml
conda activate thermosteric-slr
```

Set local paths once in the shell. These locations are examples only and must
be replaced with local data locations.

```bash
export GEBCO=/path/to/GEBCO_ocean.tif
export ANNUAL=/path/to/annual_temperature_salinity
export BASINS=/path/to/IHO_ocean_basins
export RESULTS=/path/to/annual_thermosteric_results
export VOLUME=/path/to/gebco_1deg_layer_volume_0_6000m.nc
```

## 1. Annual temperature and salinity means

Monthly input naming and the averaging method are unchanged from the analysis.

```bash
python src/annual_averaging/annual_average.py \
  --start-year 1940 --end-year 1998 \
  --temperature-root /path/to/monthly_temperature \
  --salinity-root /path/to/monthly_salinity \
  --output-dir "$ANNUAL"
```

## 2. Central 15-arc-second thermosteric calculation

This produces full-resolution NetCDF maps and downsampled JPEG figures. The
example calculates annual adjacent-year changes.

```bash
python src/thermal_expansion/run_annual_thermosteric.py 1940 2025 \
  --annual-dir "$ANNUAL" \
  --gebco "$GEBCO" \
  --output-root "$RESULTS"
```

Use `--direct` to calculate the final year relative to the first year, or
`--resume-from 2016` to continue annual processing from `2017_vs_2016`.

## 3. One-degree geometry and ensemble uncertainty

First aggregate the 15-arc-second bathymetry into depth-resolved one-degree
seawater volumes. This is a geometry preparation step, not a replacement for
the 15-arc-second central estimate.

```bash
python src/uncertainty/build_1deg_layer_volumes.py \
  --dem "$GEBCO" \
  --annual-temp "$ANNUAL/Annual_Temp_1940.nc" \
  --output-dir outputs/gebco_1deg_layer_volumes
```

Then propagate IAPv4-LE temperature-member uncertainty, holding annual central
salinity fixed, from 1940 to 2023. The script downloads required member files
from the IAP public service and retains its cache and annual states for manual
inspection or restart.

```bash
python src/uncertainty/full_spatiotemporal/run_full_uncertainty.py \
  --start-year 1940 --end-year 2023 --workers 4 \
  --annual-dir "$ANNUAL" \
  --volume-file "$VOLUME" \
  --basin-shapefile-root "$BASINS" \
  --output-dir outputs/uncertainty_1940_2023
```

## 4. Figures

The scripts require generated central results and uncertainty tables; these
files are deliberately not included in the repository. For example:

```bash
python src/figures/plot_ocean_basin_historical_partition.py \
  --results-root "$RESULTS/yearly_1940_2025" \
  --uncertainty-member-file outputs/uncertainty_1940_2023/member_results_basin.csv \
  --output-dir outputs/figures
```

## 5. Short-term projection

The retained projection workflow uses a ten-year weighted-mean baseline and a
sparse Bayesian dynamic regression. Its pre-2021 hindcasts determine the
fixed performance-weighted mixture used for 2026-2030. Store the two NOAA
driver text files (`noaa_mlo_co2_annmean.txt` and `noaa_cpc_oni.txt`) together
in a local driver directory.

```bash
export UNCERTAINTY=outputs/uncertainty_1940_2023
export DRIVERS=/path/to/noaa_driver_text_files

python src/projection/run_global_projection.py \
  --main-results "$RESULTS/yearly_1940_2025" \
  --uncertainty-results "$UNCERTAINTY" \
  --output-dir outputs/global_projection

python src/projection/run_sparse_bsts_projection.py \
  --main-results "$RESULTS/yearly_1940_2025" \
  --uncertainty-results "$UNCERTAINTY" \
  --driver-data-dir "$DRIVERS" \
  --output-dir outputs/sparse_bsts_projection

python src/projection/run_performance_weighted_ensemble.py \
  --baseline-dir outputs/global_projection \
  --bsts-dir outputs/sparse_bsts_projection \
  --output-dir outputs/performance_weighted_ensemble
```

The 2024-2025 tail-uncertainty script produces a calibrated sensitivity proxy,
not observed IAPv4-LE ensemble uncertainty. It should be labelled accordingly.

```bash
python src/uncertainty/tail_proxy/build_tail_uncertainty_proxy.py \
  --input-file "$UNCERTAINTY/uncertainty_summary_global.csv" \
  --output-dir outputs/tail_uncertainty_proxy
```

## Data and licenses

Do not redistribute third-party data in this repository unless their licences
explicitly permit it. Cite GEBCO, IAP/IAPv4-LE, IHO basin boundaries, GO-SHIP,
and any other input product according to the providers' current instructions.
