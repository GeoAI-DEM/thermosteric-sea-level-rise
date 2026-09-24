# Global Thermosteric Sea-Level Rise

This repository contains the analysis workflow used to estimate global
thermosteric sea-level rise from annual temperature, salinity, and bathymetry
data. The workflow contains four stages:

1. annual temperature and salinity averaging;
2. 15-arc-second TEOS-10 thermosteric-volume calculation;
3. 1-degree ensemble-based uncertainty quantification; and
4. short-term projection and manuscript-figure generation.

## Repository status

This repository contains code only. It contains no raw input data,
high-resolution NetCDF products, downloaded ensemble members, generated maps,
or paper results. Input and output locations are explicit command-line
arguments; machine-specific configuration files are not committed.

## Data access

Users must obtain the input data directly from their original providers and
respect the applicable licenses and citation requirements. The required inputs
are:

- GEBCO bathymetry;
- annual temperature and salinity fields;
- IAPv4-LE temperature ensemble fields for uncertainty estimation;
- IHO ocean-basin boundary shapefiles; and
- GO-SHIP gridded products where the deep-ocean residual workflow is used.

`config.example.yml` is a local path record template. The Python scripts use
the command-line paths described in [docs/RUNNING.md](docs/RUNNING.md), rather
than loading this YAML file automatically.

## Layout

```text
src/thermal_expansion/     15-arc-second TEOS-10 calculation
src/uncertainty/           1-degree volume preparation and ensemble uncertainty
src/projection/            short-term statistical projection
src/figures/               reproducible manuscript figures
config.example.yml         local input and output path template
environment.yml            tested Conda environment
```

## Reproducibility notes

- The 15-arc-second central estimate is never replaced by the 1-degree
  ensemble mean.
- Ensemble members quantify uncertainty through 2023. Later years are not
  assigned basin- or latitude-specific uncertainty without a corresponding
  ensemble product.
- The configuration in `src/thermal_expansion/run_annual_thermosteric.py`
  retains the analysed output settings: full-resolution 15-arc-second NetCDF,
  four-fold downsampled JPEG maps, and optional visual island filling.
- All published numerical tables and figures should be regenerated from a
  tagged release of this repository.

## License and citation

The public-release license and citation metadata will be added before
submission. Do not cite this staging version.
