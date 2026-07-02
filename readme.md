# 3D Baseline Lab

Experiment framework for testing baseline methods for converting 3D objects into visual collages for downstream VLM/LLM processing.

The goal of the project is to provide a simple and extensible system where different 3D preprocessing baselines can be tested on the same input model and compared through their visual outputs and metadata.

## Project idea

The framework takes a 3D object as input, normalizes it, runs a selected baseline method, and saves the result as visual collage files with JSON metadata.

```text
3D object
  -> load mesh
  -> normalize mesh
  -> run selected baseline
  -> generate rendered views
  -> assemble collage
  -> save PNG + JSON metadata
```

This makes it possible to compare different approaches to preparing 3D objects for later VLM/LLM-based processing.

## Current baselines

The framework currently supports four baseline methods:

```text
1. ISO Orthographic Baseline
2. Cap3D-style Views Baseline
3. Algorithmic Top Views Baseline
4. Isomap View Selection Baseline
```

---

## ISO Orthographic Baseline

The ISO orthographic baseline generates a standardized technical collage from a 3D object.

It renders six fixed orthographic views:

- front
- back
- left
- right
- top
- bottom

The views can be arranged using:

- first-angle projection
- third-angle projection

This baseline is useful as a simple, reproducible, engineering-style representation of a 3D object. It does not adaptively select the most informative views, but it provides a stable baseline for comparison with more advanced methods.

Example pipeline:

```text
3D object
  -> normalize mesh
  -> render six orthographic views
  -> arrange views using first-angle or third-angle projection
  -> save collage and metadata
```

Example command:

```bash
baseline-lab --input examples/models/test_part.obj --baseline iso --projection third --image-size 384 --run-name test_part_iso_third
```

---

## Cap3D-style Views Baseline

The Cap3D-style baseline is inspired by the visual rendering stage of Cap3D.

The original Cap3D pipeline uses multiple rendered views of a 3D object, applies image captioning models to these views, filters or aligns the results, and then uses an LLM to consolidate the final 3D object description.

In this repository, the implemented Cap3D-style baseline focuses only on the visual preprocessing stage:

```text
3D object
  -> normalize mesh
  -> render 8 views around the object
  -> arrange views into a collage
  -> save PNG + JSON metadata
```

This baseline does not perform caption generation yet. Captioning and LLM-based consolidation are planned as future modules.

Example command:

```bash
baseline-lab --input examples/models/test_part.obj --baseline cap3d --image-size 384 --run-name test_part_cap3d_views
```

---

## Algorithmic Top Views Baseline

The algorithmic baseline automatically selects the most informative views from a larger set of candidate renders.

It first renders 24 candidate views around the object using different azimuth and elevation angles. Then each view is scored using simple visual features:

- visible silhouette area;
- bounding box fill ratio;
- silhouette / contour complexity;
- view diversity.

The final collage is assembled from the top 8 selected views.

Example pipeline:

```text
3D object
  -> normalize mesh
  -> render 24 candidate views
  -> score each view algorithmically
  -> select top 8 diverse views
  -> assemble collage
  -> save PNG + JSON metadata
```

Example command:

```bash
baseline-lab --input examples/models/test_part.obj --baseline algorithmic --image-size 384 --run-name test_part_algorithmic_top8
```

---

## Isomap View Selection Baseline

The Isomap View Selection baseline implements a fuller multi-view pipeline.

It first renders 28 views of the object, extracts visual features from each rendered image, builds a 2D Isomap embedding, and then selects informative and diverse views from this embedding.

The 28 views are generated as:

```text
Elevation 15°: 8 azimuth views
Elevation 35°: 8 azimuth views
Elevation 60°: 8 azimuth views
Elevation 80°: 4 high-angle views

Total: 28 views
```

The selection step uses both:

- visual quality score;
- diversity in Isomap embedding space.

The visual quality score is based on:

```text
quality_score =
    0.45 * area_score
  + 0.35 * edge_score
  + 0.20 * bbox_score
```

The final view selection uses:

```text
selection_score =
    0.60 * isomap_diversity
  + 0.40 * visual_quality
```

A minimum quality threshold is used to avoid selecting visually weak side views:

```text
quality_threshold = 0.45
```

Example pipeline:

```text
3D object
  -> normalize mesh
  -> render 28 views
  -> extract image features
  -> build Isomap embedding
  -> select 3 / 4 / 6 informative diverse views
  -> assemble selected collages
  -> save images + metadata
```

Example command:

```bash
baseline-lab --input examples/models/test_part.obj --baseline isomap --image-size 256 --run-name test_part_isomap
```

This creates:

```text
outputs/
  isomap/
    test_part_isomap/
      collage.png
      all_views_collage.png
      collage_3.png
      collage_4.png
      collage_6.png
      embedding.png
      metadata.json
```

Where:

```text
all_views_collage.png — all 28 rendered views
collage_3.png         — 3 selected views
collage_4.png         — 4 selected views
collage_6.png         — 6 selected views
embedding.png         — visualization of the Isomap view embedding
metadata.json         — parameters, features, embedding coordinates and selected views
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the package in editable mode:

```bash
pip install -e .
```

The project uses:

- `trimesh` for loading and processing 3D models;
- `pyrender` for rendering views;
- `Pillow` for creating collages;
- `numpy` for numerical operations;
- `scikit-learn` for Isomap-based view selection.

---

## Demo usage

### ISO third-angle projection

```bash
baseline-lab --demo --baseline iso --projection third --image-size 384
```

This creates:

```text
outputs/
  iso/
    demo_iso_third/
      collage.png
      metadata.json
```

### ISO first-angle projection

```bash
baseline-lab --demo --baseline iso --projection first --image-size 384
```

This creates:

```text
outputs/
  iso/
    demo_iso_first/
      collage.png
      metadata.json
```

### Cap3D-style views

```bash
baseline-lab --demo --baseline cap3d --image-size 384 --run-name demo_cap3d_views
```

This creates:

```text
outputs/
  cap3d/
    demo_cap3d_views/
      collage.png
      metadata.json
```

### Algorithmic Top Views

```bash
baseline-lab --input examples/models/test_part.obj --baseline algorithmic --image-size 384 --run-name test_part_algorithmic_top8
```

This creates:

```text
outputs/
  algorithmic/
    test_part_algorithmic_top8/
      collage.png
      metadata.json
```

### Isomap View Selection

```bash
baseline-lab --input examples/models/test_part.obj --baseline isomap --image-size 256 --run-name test_part_isomap
```

This creates:

```text
outputs/
  isomap/
    test_part_isomap/
      collage.png
      all_views_collage.png
      collage_3.png
      collage_4.png
      collage_6.png
      embedding.png
      metadata.json
```

---

## Usage with a custom 3D model

Run ISO baseline on a custom model:

```bash
baseline-lab --input path/to/model.obj --baseline iso --projection third --image-size 384
```

Run Cap3D-style baseline:

```bash
baseline-lab --input path/to/model.obj --baseline cap3d --image-size 384 --run-name my_model_cap3d
```

Run Algorithmic Top Views baseline:

```bash
baseline-lab --input path/to/model.obj --baseline algorithmic --image-size 384 --run-name my_model_algorithmic
```

Run Isomap View Selection baseline:

```bash
baseline-lab --input path/to/model.obj --baseline isomap --image-size 256 --run-name my_model_isomap
```

Supported formats depend on `trimesh`, but usually include:

- `.obj`
- `.stl`
- `.ply`
- `.glb`

---

## Output structure

The framework saves each baseline result into a separate labeled folder:

```text
outputs/
  baseline_name/
    run_name/
      collage.png
      metadata.json
```

Current example structure:

```text
outputs/
  iso/
    test_part_iso_first/
      collage.png
      metadata.json

    test_part_iso_third/
      collage.png
      metadata.json

  cap3d/
    test_part_cap3d_views/
      collage.png
      metadata.json

  algorithmic/
    test_part_algorithmic_top8/
      collage.png
      metadata.json

  isomap/
    test_part_isomap/
      collage.png
      all_views_collage.png
      collage_3.png
      collage_4.png
      collage_6.png
      embedding.png
      metadata.json
```

This structure makes it easy to inspect results directly on GitHub.

---

## Metadata

Each run produces a `metadata.json` file.

Example metadata for ISO:

```json
{
  "input": "demo",
  "output": "outputs/iso/demo_iso_third/collage.png",
  "runtime_seconds": 0.327,
  "mesh": {
    "vertices": 24,
    "faces": 36,
    "extents": [1.0, 0.7, 0.8]
  },
  "result": {
    "baseline": "iso_orthographic",
    "projection": "third",
    "views": ["front", "back", "right", "left", "top", "bottom"],
    "image_size": 384
  }
}
```

Example metadata for Isomap View Selection:

```json
{
  "input": "examples/models/test_part.obj",
  "output": "outputs/isomap/test_part_isomap/collage.png",
  "runtime_seconds": 0.652,
  "mesh": {
    "vertices": 9232,
    "faces": 16260,
    "extents": [1.0, 0.912, 0.084]
  },
  "result": {
    "baseline": "isomap_view_selection",
    "total_views": 28,
    "selected_counts": [3, 4, 6],
    "image_size": 256,
    "selection": {
      "method": "quality-aware farthest point sampling in 2D Isomap embedding",
      "quality_threshold": 0.45,
      "selection_score": "0.60 * isomap_diversity + 0.40 * visual_quality"
    }
  }
}
```

The metadata file is useful for tracking:

- input source;
- output path;
- runtime;
- mesh statistics;
- baseline parameters;
- rendered views;
- feature scores;
- selected views;
- embedding coordinates.

---

## Direct output path

By default, the framework writes results into the organized `outputs/` directory.

You can still provide a direct output path with `--out`:

```bash
baseline-lab --demo --baseline iso --projection third --image-size 384 --out runs/demo_iso_third_384.png
```

This creates:

```text
runs/
  demo_iso_third_384.png
  demo_iso_third_384.json
```

The `runs/` directory is intended for local experiments and is usually ignored by Git.

---

## Repository structure

```text
3d-baseline-lab/
  README.md
  METHODS_DESCRIPTION.md
  requirements.txt
  pyproject.toml

  src/
    baseline_lab/
      __init__.py
      cli.py
      demo.py
      io.py

      baselines/
        __init__.py
        iso.py
        cap3d.py
        algorithmic.py
        isomap.py

  examples/
    models/
      test_part.obj

  outputs/
    iso/
      ...
    cap3d/
      ...
    algorithmic/
      ...
    isomap/
      ...
```

### Main files

`cli.py` handles command-line execution. It defines the input model path, baseline choice, image size, ISO projection type, run name, output root and direct output path.

`io.py` handles model loading and normalization. It loads a mesh using `trimesh`, merges scene geometries when needed, centers and scales the object, and returns a prepared mesh.

`demo.py` contains a built-in asymmetric demo object for testing the framework without an external 3D file.

The `baselines/` folder contains independent baseline implementations:

```text
iso.py          — ISO orthographic projections
cap3d.py        — fixed 8-view perspective rendering
algorithmic.py  — algorithmic top-view selection from 24 candidates
isomap.py       — 28-view rendering + Isomap-based view selection
```

---

## Current status

The framework currently supports four visual preprocessing baselines:

```text
1. ISO Orthographic Baseline
2. Cap3D-style Views Baseline
3. Algorithmic Top Views Baseline
4. Isomap View Selection Baseline
```

On the test engineering part, the baselines show different behavior:

- ISO gives a technical orthographic representation, but side views are weak for flat objects.
- Cap3D-style gives a fixed visual overview around the object.
- Algorithmic Top Views selects more informative views from 24 candidates using visual scoring.
- Isomap View Selection renders 28 views and selects 3 / 4 / 6 informative and diverse views using Isomap embedding and quality-aware filtering.

---

## Planned development

Planned next steps:

- compare mode for running all baselines on the same input model;
- automatic summary JSON for all baseline results;
- depth-based view scoring;
- captioning module for the Cap3D-style pipeline;
- VLM-based view evaluation;
- LLM/VLM-based view selection;
- automatic baseline quality comparison.

---

## Notes

The current Cap3D-style baseline is not a full reproduction of Cap3D. It only implements the rendering stage: generating several views around the object and arranging them into a collage.

The full Cap3D-like pipeline can later be extended as:

```text
3D object
  -> render multiple views
  -> caption each view
  -> filter or rank captions
  -> consolidate captions with LLM
  -> produce final 3D object description
```

The Isomap View Selection baseline is not a semantic view selection method. It selects views using image-based features, Isomap embedding and visual quality filtering. This makes the method interpretable, but it does not replace future VLM-based semantic view selection.

---

## Example commands

Run ISO third-angle demo:

```bash
baseline-lab --demo --baseline iso --projection third --image-size 384
```

Run ISO first-angle demo:

```bash
baseline-lab --demo --baseline iso --projection first --image-size 384
```

Run Cap3D-style demo:

```bash
baseline-lab --demo --baseline cap3d --image-size 384 --run-name demo_cap3d_views
```

Run ISO on a custom model:

```bash
baseline-lab --input path/to/model.obj --baseline iso --projection third --image-size 384 --run-name my_model_iso
```

Run Algorithmic Top Views on the test model:

```bash
baseline-lab --input examples/models/test_part.obj --baseline algorithmic --image-size 384 --run-name test_part_algorithmic_top8
```

Run Isomap View Selection on the test model:

```bash
baseline-lab --input examples/models/test_part.obj --baseline isomap --image-size 256 --run-name test_part_isomap
```