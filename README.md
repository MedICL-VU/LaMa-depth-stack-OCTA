# LaMa-depth-stack for Volumetric OCTA Restoration

Reference implementation for **LaMa-depth-stack**, introduced in *Reorienting 2D Inpainting for Robust Volumetric OCTA Restoration* (submitted to SPIE Medical Imaging 2027).

LaMa-depth-stack applies a frozen 2D inpainting prior to a 3D OCTA volume by changing the processing orientation. Given a volume `C` in `(B,Z,W)` order, the method independently restores each fixed-depth en face plane:

```python
plane = C[:, z, :]  # shape (B,W)
```

The restored planes are stacked along `Z`. Predictions are used only inside the supplied missing mask, so measured voxels remain exactly unchanged. The reference method performs no training, smoothing, resizing, neighboring-plane conditioning, projection optimization, or postprocessing.

## Method

For each volume, the implementation:

1. interprets nonzero mask values as missing;
2. computes `s = P99.9(C[observed & positive])` without using ground truth;
3. forms `clip(C / s, 0, 1)`;
4. rounds each `(B,W)` plane to uint8 and replicates grayscale to RGB;
5. runs frozen Big-LaMa with `255/white = missing`;
6. removes bottom/right padding returned by the backend;
7. averages output RGB channels in floating point and multiplies by `s`;
8. returns `where(mask, prediction, C)`.

Every run records the normalization, axis semantics, checkpoint hash, padding, runtime, and observed-voxel invariance check in `run_metadata.json`.

## Quickstart

### 1. Install the package

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/MedICL-VU/LaMa-depth-stack-OCTA.git
cd LaMa-depth-stack-OCTA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[lama]'
```

Install evaluation, learned-baseline, or diagnostic dependencies only when needed:

```bash
python -m pip install -e '.[evaluation,baselines,diagnostics]'
```

### 2. Prepare one volume and mask

Input files may be stored anywhere; the CLI discovers them through a CSV manifest. A minimal reconstruction directory might look like:

```text
my_experiment/
  cases.csv
  data/
    sample_corrupted.tif
    sample_mask.tif
```

Both files must have the same `(B,Z,W)` shape. `B` indexes B-scans, `Z` is axial depth, and `W` is lateral position. The mask must be nonzero/`True` where voxels are missing and zero/`False` where measurements are observed.

Create `my_experiment/cases.csv`:

```csv
case_id,corrupted_path,mask_path
sample,data/sample_corrupted.tif,data/sample_mask.tif
```

Paths may be absolute or relative to the manifest file. Ground truth is not required for reconstruction.

### 3. Provide Big-LaMa

Data, trained baseline weights, and Big-LaMa weights are not distributed. Pass a compatible Big-LaMa checkpoint with `--checkpoint` or set `LAMA_MODEL` once in your shell:

```bash
export LAMA_MODEL=/path/to/big-lama.pt
```

The reference checkpoint SHA256 is:

```text
7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c
```

Checkpoint verification is enabled by default so that inference does not silently use different weights.

### 4. Reconstruct the volume

```bash
lama-depth-stack reconstruct \
  --manifest my_experiment/cases.csv \
  --output-root my_experiment/results \
  --device cuda
```

Use `--device cpu` when CUDA is unavailable. The result is organized as:

```text
my_experiment/results/sample/
  volumes/
    raw_prediction.tif
    final_reconstruction.tif
    model_input_native_scale.tif
    missing_mask.tif
  images/
    bscans/
    fixed_depth_planes/
    backend_inputs/
    corrupted_full_axial_mip.png
    final_full_axial_mip.png
  run_metadata.json
```

`final_reconstruction.tif` is the primary output. It contains LaMa predictions only inside the missing mask and preserves observed voxels exactly. `raw_prediction.tif` retains the complete stacked model output for inspection. `run_metadata.json` records normalization, axis semantics, checkpoint provenance, padding, runtime, and the observed-voxel invariance check.

## Data and Manifests

Volumes and masks may be TIFF or NPY. A single manifest schema supports corruption generation, reconstruction, learned baselines, and evaluation; each command requires only the relevant columns.

The complete schema is:

```csv
case_id,cohort,parent_id,gt_path,corrupted_path,mask_path,full_bscan_mask_path,lateral_mask_path,corruption_metadata_path,projection_kind,layer_path,axial_offset
case01,inhouse,acquisition01,data/gt.tif,data/corrupted.tif,data/mask.tif,data/full_mask.tif,data/lateral_mask.tif,,full_axial,,
```

Only `case_id`, `corrupted_path`, and `mask_path` are required for reconstruction. Evaluation additionally requires `gt_path`, `full_bscan_mask_path`, `lateral_mask_path`, and `projection_kind`.

- Use `projection_kind=full_axial` for full-depth projection.
- Use `projection_kind=ilm_opl` for OCTA-500 layer-guided projection. This also requires an NPY, NPZ, or MAT `layer_path` containing at least ILM and OPL surfaces in `(surface,B,W)` order.
- Set the optional zero-based `axial_offset` when layer surfaces refer to an uncropped source volume.
- Set `parent_id` when multiple in-house subvolumes should be aggregated as one acquisition.

## Additional Commands

Generate reproducible synthetic corruptions from clean volumes or run the linear-interpolation baseline:

```bash
lama-depth-stack corrupt --manifest clean_cases.csv --output-root data/corrupted
lama-depth-stack linear --manifest cases.csv --output-root results/linear
```

For corruption generation, the input manifest requires `case_id`, `gt_path`, and `projection_kind`. The command writes corrupted volumes, component masks, metadata, and a ready-to-use output manifest.

Evaluate any method using either its standard output directory or a CSV with `case_id,prediction_path` columns:

```bash
lama-depth-stack evaluate \
  --manifest cases.csv \
  --predictions results/lama_depth_stack \
  --output-dir results/evaluation
```

The evaluator applies missing-only paste-back to every method, divides each prediction and GT by that case's GT maximum, and uses disjoint corruption supports. Grad L1 and AlexNet LPIPS are evaluated on fully missing B-scans with lateral overlap excluded. MIP L1 and Pearson NCC are evaluated after full-depth or ILM--OPL projection. In-house subvolumes are first averaged by `parent_id`; OCTA-500 3 mm and 6 mm volumes remain independent units. Per-case, independent-unit, and cohort summaries are written as CSV files.

## Baselines

The repository includes three comparison implementations:

- mask-aware linear interpolation along `B`;
- supervised VAMOS-OCTA with a 9-B-scan corrupted input window;
- SOAD blind-slice with a 7-B-scan input window.

```bash
lama-depth-stack train-vamos --train-manifest train.csv --validation-manifest val.csv --output-dir runs/vamos --device cuda
lama-depth-stack predict-vamos --manifest test.csv --checkpoint runs/vamos/checkpoints/best.pt --output-root results/vamos --device cuda

lama-depth-stack train-soad --train-manifest train.csv --validation-manifest val.csv --output-dir runs/soad --device cuda
lama-depth-stack predict-soad --manifest test.csv --checkpoint runs/soad/checkpoints/best.pt --output-root results/soad --device cuda
```

Typed dataclasses define each baseline's default configuration. An optional strict JSON file may override exposed fields through `--config`; no configuration file is required for the defaults. Training consumes clean targets only where required by the baseline objective. Prediction never loads clean targets, and all final outputs use missing-only paste-back.

## Analyses from the Paper

Two manifest-driven analyses reproduce the diagnostic logic used in the paper without embedding its datasets or numerical results:

```bash
# OCTA-500 3 mm nested gaps at L={1,2,4,6,9,12} and three fixed locations.
lama-depth-stack diagnostic-nested-prepare --manifest octa500_3mm.csv --output-root nested
lama-depth-stack diagnostic-nested-analyze \
  --manifest nested/manifest.csv \
  --conditions nested/nested_conditions.csv \
  --predictions LaMa=results/lama \
  --predictions VAMOS=results/vamos \
  --output-dir nested/analysis

# Adjacent fixed-depth-plane NCC and the separate 5x5 per-B-scan median diagnostic.
lama-depth-stack diagnostic-median-prepare --manifest inhouse.csv --output-root median
lama-depth-stack diagnostic-consistency --manifest inhouse.csv --output-dir consistency
```

Both workflows save machine-readable CSV summaries and high-resolution plots. Learned-baseline runs additionally store `run_config.json`, epoch-level CSV histories, and reloadable best/last checkpoints.

## Repository Structure

```text
src/lama_depth_stack/
  cli.py                 command-line interface
  corruption.py          synthetic corruption protocol
  core/                  I/O, manifests, normalization, backend, reconstruction
  evaluation/            projections, disjoint supports, metrics, aggregation
  baselines/             Linear, VAMOS-OCTA, and SOAD
  diagnostics/           nested-gap and consistency analyses
```

## License and Acknowledgments

This repository is released under the MIT License. LaMa-depth-stack uses the [Big-LaMa](https://github.com/advimman/lama) model through [`simple-lama-inpainting`](https://github.com/enesmsahin/simple-lama-inpainting). The VAMOS-OCTA and SOAD baseline modules adapt MIT-licensed code and retain their upstream notices directly in `baselines/vamos_octa.py` and `baselines/soad.py`. Both methods should be cited when used as comparisons.
