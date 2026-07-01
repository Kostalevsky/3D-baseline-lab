import argparse
import json
from pathlib import Path
from time import perf_counter

from baseline_lab.baselines.iso import IsoOrthographicBaseline
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
        choices=["iso"],
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
        "--out",
        type=str,
        default="runs/output.png",
        help="Output collage path",
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
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    result = baseline.run(mesh)

    output_path = Path(args.out)
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

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved collage: {output_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()