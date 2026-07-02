import argparse
import json
from pathlib import Path
from time import perf_counter

from baseline_lab.baselines.iso import IsoOrthographicBaseline
from baseline_lab.baselines.cap3d import Cap3DViewsBaseline
from baseline_lab.baselines.algorithmic import AlgorithmicTopViewsBaseline
from baseline_lab.baselines.isomap import IsomapViewSelectionBaseline
from baseline_lab.demo import create_demo_mesh
from baseline_lab.io import load_mesh, normalize_mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D baseline experiment framework"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input 3D model: .obj, .stl, .ply, .glb",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use built-in demo mesh instead of input file",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default="iso",
        choices=["iso", "cap3d", "algorithmic", "isomap"],
        help="Baseline method to run",
    )

    parser.add_argument(
        "--projection",
        type=str,
        default="third",
        choices=["first", "third"],
        help="ISO projection method",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Size of each rendered view in pixels",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for organized baseline outputs",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the output run folder",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional direct output collage path. If not provided, output-root/baseline/run-name/collage.png is used.",
    )

    args = parser.parse_args()

    if not args.demo and args.input is None:
        raise ValueError("Provide --input path/to/model.obj or use --demo")

    start_time = perf_counter()

    if args.demo:
        mesh = create_demo_mesh()
        input_source = "demo"
    else:
        mesh = load_mesh(args.input)
        input_source = args.input

    mesh = normalize_mesh(mesh)

    if args.baseline == "iso":
        baseline = IsoOrthographicBaseline(
            projection=args.projection,
            image_size=args.image_size,
        )
    elif args.baseline == "cap3d":
        baseline = Cap3DViewsBaseline(
            image_size=args.image_size,
            num_views=8,
        )
    elif args.baseline == "algorithmic":
        baseline = AlgorithmicTopViewsBaseline(
            image_size=args.image_size,
        )
    elif args.baseline == "isomap":
        baseline = IsomapViewSelectionBaseline(
            image_size=args.image_size,
    )
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    result = baseline.run(mesh)

    # Build output path.
    # If --out is provided, save directly to that path.
    # Otherwise, use organized structure:
    # outputs/{baseline}/{run_name}/collage.png
    if args.out is not None:
        output_path = Path(args.out)
    else:
        if args.run_name is not None:
            run_name = args.run_name
        else:
            if args.demo:
                input_stem = "demo"
            else:
                input_stem = Path(args.input).stem

            if args.baseline == "iso":
                run_name = f"{input_stem}_{args.baseline}_{args.projection}"
            else:
                run_name = f"{input_stem}_{args.baseline}"

        output_path = Path(args.output_root) / args.baseline / run_name / "collage.png"

    baseline.save(result, output_path)

    runtime_seconds = perf_counter() - start_time

    metadata = {
        "input": input_source,
        "output": str(output_path),
        "runtime_seconds": runtime_seconds,
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "extents": mesh.extents.tolist(),
        },
        "result": result.metadata,
    }

    # If user gave a direct --out path, metadata goes next to it with .json suffix.
    # Otherwise, metadata goes inside the same organized run folder.
    if args.out is not None:
        metadata_path = output_path.with_suffix(".json")
    else:
        metadata_path = output_path.parent / "metadata.json"

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved collage: {output_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()