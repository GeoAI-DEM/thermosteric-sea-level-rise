# Manuscript Workflow

## Scope and Status

`src/manuscript` preserves the recent study scripts with their formulas and
statistical functions unchanged. Only input/output paths and import setup
were adapted. Dated directory names identify calculation stages, not release
dates. `workflow_snapshot_manifest.json` records source-relative filenames,
original and public-copy SHA-256 hashes, and AST checks.

The archive complements the older configurable entrypoints under `src/`.
Do not interchange output schemas or assume an older preview is the selected
manuscript figure. The Chinese [method record](METHODS.md) explains the actual
implementation, uncertainty definitions and outstanding provenance questions.

## Environment

The original environment used Python 3.7, NumPy 1.21.6, pandas 1.3.5,
SciPy 1.7.3, Matplotlib 3.5.3, scikit-learn 1.0.2, netCDF4 1.6.5,
GSW 3.4.0 and PyTorch 1.8.1+cu111. The Conda specification is a legacy
dependency record, not a solved lockfile or a modern-Python compatibility
guarantee. PyTorch/CUDA and optional GPU dependencies must be installed
separately for the machine. Inspect the imports of the selected entrypoint;
offline statistical tests do not require GPU execution.

```bash
conda env create -f environment.yml
conda activate thermosteric-slr
python -B tools/run_tests.py
```

GDAL/PROJ/Cartopy require compatible native libraries and map assets.
Cartopy may need its Natural Earth data installed before offline plotting.
No global data downloads or GPU tests are performed by the test command.

## Workspace and Inputs

Run commands from the repository root. Keep study data outside source code:

```bash
export WORK=/path/to/study_workspace
export THERMOSTERIC_WORKSPACE="$WORK"
export THERMOSTERIC_ANNUAL_DIR=/path/to/annual_averages
export THERMOSTERIC_GEBCO=/path/to/GEBCO_ocean.tif
export THERMOSTERIC_BASINS=/path/to/IHO_ocean_basins
python -B src/manuscript/run.py list
```

If the three input overrides are omitted, their defaults are
`$WORK/inputs/annual_averages`, `$WORK/inputs/GEBCO_ocean.tif` and
`$WORK/inputs/IHO_ocean_basins`. Without a workspace setting the default is
`outputs` relative to the current directory. Always set it explicitly for a
real study. The launcher changes the working directory to that workspace;
relative script arguments are therefore workspace-relative.

Important workspace products expected by the archived scripts include:

```text
Results/yearly_1940_2025/<target>_vs_<base>/
uncertainty_preprocessing/gebco_1deg_layer_volumes/
  gebco_1deg_layer_volume_0_6000m.nc
uncertainty_analysis/full_spatiotemporal_uncertainty/
  results_1940_2023_lon_aligned/
contribution_structure_analysis/step1_long_term_20261014/
contribution_structure_analysis/step2_interannual_20261014/
contribution_structure_analysis/nearest_basin_reassignment_20261015/
contribution_structure_analysis/step3_nearest_20261015/
figures/
```

Third-party inputs, local result CSVs, full-resolution NCs and rendered
figures are intentionally not bundled. Source-only checkout cannot produce
the final paper figures without preparing these products. Download scripts
use provider URLs recorded in source; obtain appropriate access permissions
and inspect their options before starting a large download.

## Execution

Use the launcher for archived scripts, not direct execution of the file:

```bash
python -B src/manuscript/run.py --workspace "$WORK" \
  contribution_structure_analysis/calculate_long_term_contributions.py \
  --input-root "$WORK/Results/yearly_1940_2025" \
  --output-dir "$WORK/contribution_structure_analysis/step1_long_term_20261014"

python -B src/manuscript/run.py --workspace "$WORK" \
  contribution_structure_analysis/calculate_interannual_variability.py \
  --input-dir "$WORK/contribution_structure_analysis/step1_long_term_20261014" \
  --output-dir "$WORK/contribution_structure_analysis/step2_interannual_20261014"

python -B src/manuscript/run.py --workspace "$WORK" \
  contribution_structure_analysis/en4_uniform_recalculation/run_en4_uniform_comparison.py \
  --help
```

These calculations require the previous stage's files. EN4 has a separate
download/preparation stage; its common mask must be frozen over the stated
full period before comparing subperiods. Recent basin completion also reads
large NC/DEM files and is not a cheap metadata operation.

For temperature members use the explicit-path command in `RUNNING.md`.
The default depth convention remains `--depth-binning legacy`; the optional
`main_compatible` assigns boundary depths to the shallower band. Use a new
output directory when changing the rule, source alignment or geometry;
never merge caches/results from incompatible settings. Existing archives
without the metadata field cannot be assumed to use the newer convention.
Neither option makes coarse-grid thermodynamics identical to 15-arcsecond
thermodynamics.

## Figure and Experiment Map

Paths below are relative to `src/manuscript`; launch them as above. Some
scripts have no CLI and read the staged workspace products directly.

| Paper item | Entrypoint / evidence stage |
|---|---|
| Figure 01, introduction | `figures/introduction_logic_redesign_20261003/plot_introduction_right_column.py` |
| Figures 02/03, global and depth evolution | `figures/rebuild_lon_aligned_figures.py` and `figures/plot_thermal_expansion_with_uncertainty.py` |
| Figure 04, spatial volume changes | Final direct-map input/endpoint provenance still needs confirmation; exact regeneration is not certified here |
| Figure 05, historical basins | `figures/plot_ocean_basin_historical_partition.py` |
| Figure 06, latitude shells | `figures/aggregate_15arcsec_nested_latitude.py`, then `figures/plot_latitude_shell_ridgelines_2025.py` |
| Figure 07, interannual allocation | `contribution_structure_analysis/nearest_basin_reassignment_20261015/`, then `figures/step2_interannual_2025_nearest_20261015/plot_nearest_basin_variability.py` |
| Figure S04, robustness | `contribution_structure_analysis/step3_nearest_20261015/calculate_figure07_robustness.py`, then `figures/step3_robustness_nearest_20261015/plot_robustness.py` |
| Figure 08, S05/S06, comparison tables | `contribution_structure_analysis/compare_*`, `contribution_structure_analysis/en4_uniform_recalculation/`, then `figures/step4_cross_product_preview_20261015/plot_cross_product_figures.py` |
| Layer heatmaps and S03 | `figures/plot_thermal_expansion_with_uncertainty.py` |
| Tail proxies / conditional prediction | `uncertainty_analysis/tail_uncertainty_proxy/` and `short_term_projection/` |

Figure exports and verification helpers remain in the same archived folders.
Keep alternative designs separate from the chosen final exports. The code
snapshot includes intermediate dependencies for provenance; inclusion does
not designate every preview as a final result.

## Scientific Boundaries

- Source-field hole filling can change numerical results; it is not only
  visual filling. DEM volume still determines where water exists.
- Temperature-member SD, conditional block-bootstrap intervals, tail proxies
  and prediction intervals answer different questions.
- Annual member paths preserve covariance when accumulating uncertainty;
  summing annual variances is not a substitute for this calculation.
- Basin nearest-neighbor completion applies to the stated recent-period
  totals, not silently to historical basin-depth fields or member masks.
- The latitude shells combine both hemispheres and cover 86 S to 90 N.
- The model called BSTS is an approximation. External inputs and prediction
  uncertainty are conditional; these are not emissions-scenario forecasts.
- Cross-product agreement is not independent truth and does not validate the
  deep ocean. See `METHODS.md` for salinity, pressure, layer and mask caveats.

## Updating and Archiving

Maintainers can refresh the curated source copy with
`python -B tools/sync_manuscript_snapshot.py --source-root /path/to/study --refresh`.
This is an explicit snapshot refresh, not a user execution requirement.
Review path transformations and rerun tests after refreshing. The tool checks
source bytes were not altered and compares numerical function ASTs.

Before submission, confirm unresolved method provenance, select a software
license, review the public files, and archive a tagged commit with a DOI.
Generated result data require their own provenance and distribution decision.
