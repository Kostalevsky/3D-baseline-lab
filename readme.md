# 3D Baseline Lab

Experiment framework for testing baseline methods for converting 3D objects into visual collages for downstream VLM/LLM processing.

## Current baseline

### ISO Orthographic Baseline

The first implemented baseline generates a standardized orthographic collage from a 3D object.

It renders six fixed views:

- front
- back
- left
- right
- top
- bottom

The views can be arranged using:

- first-angle projection
- third-angle projection

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .


## Demo usage

```bash
baseline-lab --demo --baseline iso --projection third --image-size 384 --out runs/demo_iso_third_384.png
baseline-lab --demo --baseline iso --projection first --image-size 384 --out runs/demo_iso_first_384.png

Usage with a custom 3D model
```bash
baseline-lab --input path/to/model.obj --baseline iso --projection third --image-size 384 --out runs/model_iso.png