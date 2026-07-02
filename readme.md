# 3D Baseline Lab

Experiment framework for testing baseline methods for converting 3D objects into visual collages for downstream VLM/LLM processing.

The goal of the project is to provide a simple and extensible system where different 3D preprocessing baselines can be tested on the same input model and compared through their visual outputs and metadata.

## Project idea

The framework takes a 3D object as input, normalizes it, runs a selected baseline method, and saves the result as a visual collage.

```text
3D object
  -> load mesh
  -> normalize mesh
  -> run selected baseline
  -> generate collage
  -> save PNG + JSON metadata
```

This makes it possible to compare different approaches to preparing 3D objects for later VLM/LLM-based processing.

## Current baselines

### ISO Orthographic Baseline

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

### Cap3D-style Views Baseline

The Cap3D-style baseline is inspired by the visual rendering stage of Cap3D.

The original Cap3D pipeline uses multiple rendered views of a 3D object, applies image captioning models to these views, filters or aligns the results, and then uses an LLM to consolidate the final 3D object description.

In this repository, the first implemented Cap3D-style baseline focuses only on the visual preprocessing stage:

```text
3D object
  -> normalize mesh
  -> render 8 views around the object
  -> arrange views into a collage
  -> save PNG + JSON metadata
```

This baseline does not perform caption generation yet. Captioning and LLM-based consolidation are planned as future modules.

### Algorithmic Top Views Baseline

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

Usage:

```bash
baseline-lab --input examples/models/test_part.obj --baseline algorithmic --image-size 384 --run-name test_part_algorithmic_top8
```

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

### Algorithmic Top Views Baseline

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

## Usage with a custom 3D model

```bash
baseline-lab --input path/to/model.obj --baseline iso --projection third --image-size 384
```

The output will be saved automatically to:

```text
outputs/
  iso/
    model_iso_third/
      collage.png
      metadata.json
```

You can also specify a custom run name:

```bash
baseline-lab --input path/to/model.obj --baseline iso --projection third --image-size 384 --run-name my_model_iso
```

Output:

```text
outputs/
  iso/
    my_model_iso/
      collage.png
      metadata.json
```

Supported formats depend on `trimesh`, but usually include:

- `.obj`
- `.stl`
- `.ply`
- `.glb`

## Output structure

The framework saves each baseline result into a separate labeled folder:

```text
outputs/
  baseline_name/
    run_name/
      collage.png
      metadata.json
```

Example:

```text
outputs/
  iso/
    demo_iso_third/
      collage.png
      metadata.json

  cap3d/
    demo_cap3d_views/
      collage.png
      metadata.json
```

This structure makes it easy to inspect results directly on GitHub.

## Metadata

Each run produces a `metadata.json` file.

Example:

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

The metadata file is useful for tracking:

- input source
- output path
- runtime
- mesh statistics
- baseline parameters
- rendered views

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

## Repository structure

```text
3d-baseline-lab/
  README.md
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
  outputs/
    iso/
      ...
    cap3d/
      ...
  examples/
```

## Planned baselines

Planned future methods:

- fixed multi-view rendering
- geometric view selection
- view scoring based on silhouette area and contour complexity
- Cap3D-style captioning pipeline
- VLM-based view selection
- LLM/VLM comparison mode
- automatic baseline quality comparison

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