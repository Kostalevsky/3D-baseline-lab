from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Cap3DResult:
    collage: Image.Image
    metadata: dict


class Cap3DViewsBaseline:
    """
    Cap3D-style visual baseline.

    Это не полный Cap3D captioning pipeline.
    Сейчас мы реализуем только визуальную часть:
    3D object -> 8 rendered views -> collage.

    Полный Cap3D дополнительно включает:
    - image captioning для каждого вида;
    - image-text alignment / filtering;
    - LLM consolidation в итоговое описание объекта.
    """

    def __init__(
        self,
        image_size: int = 384,
        num_views: int = 8,
        elevation_deg: float = 25.0,
        radius: float = 2.5,
        background_color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.image_size = image_size
        self.num_views = num_views
        self.elevation_deg = elevation_deg
        self.radius = radius
        self.background_color = background_color

    def run(self, mesh: trimesh.Trimesh) -> Cap3DResult:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_size,
            viewport_height=self.image_size,
        )

        try:
            views = {}
            for i in range(self.num_views):
                azimuth_deg = 360.0 * i / self.num_views
                eye = self._camera_position(azimuth_deg, self.elevation_deg)
                views[f"view_{i:02d}_{int(azimuth_deg)}deg"] = self._render_view(
                    mesh=mesh,
                    renderer=renderer,
                    eye=eye,
                )
        finally:
            renderer.delete()

        collage = self._make_collage(views)

        metadata = {
            "baseline": "cap3d_views",
            "description": "Cap3D-style multi-view rendering baseline without captioning",
            "num_views": self.num_views,
            "elevation_deg": self.elevation_deg,
            "radius": self.radius,
            "image_size": self.image_size,
            "views": list(views.keys()),
        }

        return Cap3DResult(collage=collage, metadata=metadata)

    def save(self, result: Cap3DResult, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.collage.save(output_path)

    def _camera_position(self, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)

        x = self.radius * math.cos(elevation) * math.sin(azimuth)
        y = -self.radius * math.cos(elevation) * math.cos(azimuth)
        z = self.radius * math.sin(elevation)

        return np.array([x, y, z], dtype=float)

    def _render_view(
        self,
        mesh: trimesh.Trimesh,
        renderer: pyrender.OffscreenRenderer,
        eye: np.ndarray,
    ) -> Image.Image:
        scene = pyrender.Scene(
            bg_color=[*self.background_color, 255],
            ambient_light=[0.45, 0.45, 0.45],
        )

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.45, 0.45, 0.45, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.65,
        )

        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene.add(render_mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

        camera_pose = self._look_at(
            eye=eye,
            target=np.array([0.0, 0.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
        )

        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
        scene.add(light, pose=camera_pose)

        color, _ = renderer.render(scene)
        return Image.fromarray(color)

    def _look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        true_up = np.cross(right, forward)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = true_up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye

        return pose

    def _make_collage(self, views: dict[str, Image.Image]) -> Image.Image:
        cols = 4
        rows = math.ceil(len(views) / cols)

        label_height = 38
        gap = 16

        tile_width = self.image_size
        tile_height = self.image_size + label_height

        canvas_width = cols * tile_width + (cols - 1) * gap
        canvas_height = rows * tile_height + (rows - 1) * gap

        collage = Image.new(
            "RGB",
            (canvas_width, canvas_height),
            self.background_color,
        )

        for idx, (name, image) in enumerate(views.items()):
            col = idx % cols
            row = idx // cols

            x = col * (tile_width + gap)
            y = row * (tile_height + gap)

            tile = self._add_label(image, name.upper())
            collage.paste(tile, (x, y))

        return collage

    def _add_label(self, image: Image.Image, label: str) -> Image.Image:
        label_height = 38

        tile = Image.new(
            "RGB",
            (self.image_size, self.image_size + label_height),
            self.background_color,
        )

        tile.paste(image, (0, 0))

        draw = ImageDraw.Draw(tile)

        draw.rectangle(
            [0, 0, self.image_size - 1, self.image_size - 1],
            outline=(210, 210, 210),
            width=2,
        )

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

        draw.text((12, self.image_size + 10), label, fill=(0, 0, 0), font=font)

        return tile