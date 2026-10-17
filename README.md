# Global Thermosteric Sea-Level Rise

This repository contains the analysis workflow used to estimate global
thermosteric sea-level rise from annual temperature, salinity, and bathymetry
data. The workflow contains these stages:

1. annual temperature and salinity averaging;
2. 15-arc-second TEOS-10 thermosteric-volume calculation;
3. 1-degree ensemble-based uncertainty quantification; and
4. depth, basin and latitude contribution analysis;
5. interannual variance/covariance and joint block-bootstrap sensitivity;
6. matched comparisons with NOAA, Ishii, EN4, Copernicus and CSIRO; and
7. short-term statistical prediction and manuscript-figure generation.

## Repository status

This repository contains code only. It contains no raw input data,
high-resolution NetCDF products, downloaded ensemble members, generated maps,
or paper results. Input and output locations are explicit command-line
arguments or workspace-relative paths; machine-specific configuration files
are not committed. The latest manuscript workflow is a path-adapted source
snapshot, not a claim of independent physical validation or a frozen release.

Start with [the manuscript workflow](docs/MANUSCRIPT_WORKFLOW.md),
[implementation methods and limitations](docs/METHODS.md), and
[the original CLI guide](docs/RUNNING.md).

## Data access

Users must obtain the input data directly from their original providers and
respect the applicable licenses and citation requirements. The required inputs
are:

- GEBCO bathymetry;
- annual temperature and salinity fields;
- IAPv4-LE temperature ensemble fields for uncertainty estimation;
- IHO ocean-basin boundary shapefiles; and
- reference-product data for the corresponding comparison experiments.

The older GO-SHIP residual script is retained for provenance; it is not the
principal temperature-ensemble uncertainty workflow.

`config.example.yml` is a local path record template. The Python scripts use
the command-line paths described in [docs/RUNNING.md](docs/RUNNING.md), rather
than loading this YAML file automatically.

## Layout

```text
src/thermal_expansion/     15-arc-second TEOS-10 calculation
src/uncertainty/           1-degree volume preparation and ensemble uncertainty
src/projection/            short-term statistical projection
src/figures/               original figure CLI entrypoints
src/manuscript/            recent analyses, comparisons, figures and source tests
tools/run_tests.py         offline numerical and portability tests
docs/METHODS.md            implementation record and scientific limitations
config.example.yml         local input and output path template
environment.yml            legacy dependency specification (not a lockfile)
```

## Reproducibility notes

- The 15-arc-second central estimate is never replaced by the 1-degree
  ensemble mean.
- Ensemble members quantify uncertainty through 2023. Later years are not
  assigned basin- or latitude-specific uncertainty without a corresponding
  ensemble product.
- The configuration in `src/thermal_expansion/run_annual_thermosteric.py`
  retains the analysed output settings: full-resolution 15-arc-second NetCDF,
  four-fold downsampled JPEG maps, and optional source-field hole filling.
  Filled temperature/salinity values participate in the calculation; this is
  not purely cosmetic filling. The DEM wet-cell volume still excludes land.
- A 15-arc-second integration grid does not provide independent temperature
  or salinity information at that resolution.
- The short-term model called BSTS in script names is an approximate sparse
  Bayesian regression with dynamic residuals, not a full Bayesian structural
  time-series posterior. Predictions are conditional statistical outlooks.
- Archived member uncertainty and the central estimate have documented depth,
  mask and normalization differences; do not call them a complete matched
  observational confidence interval.
- All published numerical tables and figures should be regenerated from a
  tagged release of this repository.

## Quick Check

```bash
python -B tools/run_tests.py
python -B src/manuscript/run.py list
```

Tests use small synthetic inputs and do not download data or integrate the
global DEM. They do not certify every final figure or scientific assumption.

## License and Citation

Repository citation metadata is in [CITATION.cff](CITATION.cff). For manuscript
reproducibility, record the exact commit and archive a release with a DOI when
ready. A software license has not yet been selected; public visibility alone
does not grant a reuse license. No paper DOI or software DOI is asserted here.
