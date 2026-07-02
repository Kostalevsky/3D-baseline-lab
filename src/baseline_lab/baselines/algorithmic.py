from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont


@dataclass
class AlgorithmicResult:
    collage: Image.Image
    metadata: dict


class AlgorithmicTopViewsBaseline:
    """
    Algorithmic top-views baseline.

    Логика метода:
    1. Рендерим много candidate views вокруг объекта.
    2. Для каждого вида считаем простые визуальные признаки:
       - silhouette area;
       - bounding box fill;
       - edge / contour complexity.
    3. Выбираем top-k информативных и разнообразных видов.
    4. Собираем итоговый collage.
    """

    def __init__(
        self,
        image_size: int = 384,
        candidate_azimuths: int = 8,
        candidate_elevations: tuple[float, ...] = (15.0, 35.0, 60.0),
        top_k: int = 8,
        radius: float = 2.5,
        diversity_weight: float = 0.15,
        background_color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.image_size = image_size
        self.candidate_azimuths = candidate_azimuths
        self.candidate_elevations = candidate_elevations
        self.top_k = top_k
        self.radius = radius
        self.diversity_weight = diversity_weight
        self.background_color = background_color

    def run(self, mesh: trimesh.Trimesh) -> AlgorithmicResult:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_size,
            viewport_height=self.image_size,
        )

        try:
            candidates = self._render_candidates(mesh, renderer)
        finally:
            renderer.delete()

        selected = self._select_diverse_top_views(candidates)
        collage = self._make_collage(selected)

        metadata = {
            "baseline": "algorithmic_top_views",
            "description": "Algorithmic view selection based on silhouette area, bounding box fill, contour complexity, and view diversity",
            "candidate_views": len(candidates),
            "selected_views": len(selected),
            "top_k": self.top_k,
            "candidate_azimuths": self.candidate_azimuths,
            "candidate_elevations": list(self.candidate_elevations),
            "radius": self.radius,
            "image_size": self.image_size,
            "scoring": {
                "area_score": "normalized visible object area",
                "bbox_fill_score": "object mask area divided by its 2D bounding box area",
                "edge_score": "normalized silhouette/contour complexity",
                "final_score": "0.45 * area_score + 0.35 * edge_score + 0.20 * bbox_fill_score",
                "diversity": "greedy selection with angular diversity bonus",
            },
            "selected": [
                self._candidate_metadata(candidate, rank=index)
                for index, candidate in enumerate(selected)
            ],
            "all_candidates": [
                self._candidate_metadata(candidate, rank=None)
                for candidate in candidates
            ],
        }

        return AlgorithmicResult(collage=collage, metadata=metadata)

    def save(self, result: AlgorithmicResult, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.collage.save(output_path)

    def _render_candidates(
        self,
        mesh: trimesh.Trimesh,
        renderer: pyrender.OffscreenRenderer,
    ) -> list[dict]:
        candidates = []

        index = 0

        for elevation_deg in self.candidate_elevations:
            for azimuth_index in range(self.candidate_azimuths):
                azimuth_deg = 360.0 * azimuth_index / self.candidate_azimuths

                eye = self._camera_position(
                    azimuth_deg=azimuth_deg,
                    elevation_deg=elevation_deg,
                )

                image = self._render_view(
                    mesh=mesh,
                    renderer=renderer,
                    eye=eye,
                )

                score_data = self._compute_view_score(image)

                candidate = {
                    "index": index,
                    "name": f"view_{index:02d}_az{int(azimuth_deg):03d}_el{int(elevation_deg):02d}",
                    "azimuth_deg": azimuth_deg,
                    "elevation_deg": elevation_deg,
                    "eye": eye,
                    "image": image,
                    **score_data,
                }

                candidates.append(candidate)
                index += 1

        return candidates

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

    def _compute_view_score(self, image: Image.Image) -> dict:
        """
        Считает score одного rendered view.

        Мы отделяем объект от белого фона простым threshold:
        если пиксель заметно темнее белого, считаем его частью объекта.
        """
        arr = np.asarray(image.convert("RGB")).astype(np.float32)

        # Маска объекта: всё, что не похоже на белый фон.
        mask = np.any(arr < 245, axis=2)

        height, width = mask.shape
        image_area = height * width

        object_pixels = int(mask.sum())

        if object_pixels == 0:
            return {
                "area_ratio": 0.0,
                "area_score": 0.0,
                "bbox_fill_score": 0.0,
                "edge_ratio": 0.0,
                "edge_score": 0.0,
                "base_score": 0.0,
            }

        area_ratio = object_pixels / image_area

        ys, xs = np.where(mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
        bbox_fill_score = object_pixels / bbox_area

        edges = np.zeros_like(mask, dtype=bool)
        edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        edges[1:, :] |= mask[1:, :] != mask[:-1, :]

        edge_ratio = edges.sum() / image_area

        # Нормализация простая и объяснимая:
        # area_score: хорошо, когда объект занимает заметную часть кадра;
        # edge_score: хорошо, когда у силуэта/отверстий много контуров.
        area_score = min(area_ratio / 0.30, 1.0)
        edge_score = min(edge_ratio / 0.04, 1.0)

        base_score = (
            0.45 * area_score
            + 0.35 * edge_score
            + 0.20 * bbox_fill_score
        )

        return {
            "area_ratio": float(area_ratio),
            "area_score": float(area_score),
            "bbox_fill_score": float(bbox_fill_score),
            "edge_ratio": float(edge_ratio),
            "edge_score": float(edge_score),
            "base_score": float(base_score),
        }

    def _select_diverse_top_views(self, candidates: list[dict]) -> list[dict]:
        """
        Greedy selection:
        1. Первый вид — максимальный base_score.
        2. Каждый следующий вид получает бонус за угловое отличие от уже выбранных.
        """
        if not candidates:
            return []

        remaining = candidates.copy()
        selected = []

        while remaining and len(selected) < self.top_k:
            best_candidate = None
            best_adjusted_score = -1.0

            for candidate in remaining:
                if not selected:
                    diversity_bonus = 0.0
                else:
                    diversity_bonus = self._min_angular_distance_to_selected(
                        candidate,
                        selected,
                    )

                adjusted_score = candidate["base_score"] + self.diversity_weight * diversity_bonus

                if adjusted_score > best_adjusted_score:
                    best_adjusted_score = adjusted_score
                    best_candidate = candidate

            best_candidate = best_candidate.copy()
            best_candidate["adjusted_score"] = float(best_adjusted_score)

            selected.append(best_candidate)
            remaining = [
                candidate
                for candidate in remaining
                if candidate["index"] != best_candidate["index"]
            ]

        return selected

    def _min_angular_distance_to_selected(
        self,
        candidate: dict,
        selected: list[dict],
    ) -> float:
        candidate_dir = candidate["eye"] / np.linalg.norm(candidate["eye"])

        distances = []

        for selected_candidate in selected:
            selected_dir = selected_candidate["eye"] / np.linalg.norm(selected_candidate["eye"])

            cosine = np.dot(candidate_dir, selected_dir)
            cosine = np.clip(cosine, -1.0, 1.0)

            angle = math.acos(cosine)

            # Нормализуем: 0 = тот же ракурс, 1 = противоположный ракурс.
            normalized_distance = angle / math.pi
            distances.append(normalized_distance)

        return float(min(distances))

    def _make_collage(self, selected: list[dict]) -> Image.Image:
        cols = 4
        rows = math.ceil(len(selected) / cols)

        label_height = 42
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

        for idx, candidate in enumerate(selected):
            col = idx % cols
            row = idx // cols

            x = col * (tile_width + gap)
            y = row * (tile_height + gap)

            label = (
                f"AZ{int(candidate['azimuth_deg']):03d} "
                f"EL{int(candidate['elevation_deg']):02d} "
                f"S{candidate['base_score']:.2f}"
            )

            tile = self._add_label(candidate["image"], label)
            collage.paste(tile, (x, y))

        return collage

    def _add_label(self, image: Image.Image, label: str) -> Image.Image:
        label_height = 42

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

        draw.text((12, self.image_size + 11), label, fill=(0, 0, 0), font=font)

        return tile

    def _candidate_metadata(self, candidate: dict, rank: int | None) -> dict:
        data = {
            "index": candidate["index"],
            "name": candidate["name"],
            "azimuth_deg": candidate["azimuth_deg"],
            "elevation_deg": candidate["elevation_deg"],
            "area_ratio": candidate["area_ratio"],
            "area_score": candidate["area_score"],
            "bbox_fill_score": candidate["bbox_fill_score"],
            "edge_ratio": candidate["edge_ratio"],
            "edge_score": candidate["edge_score"],
            "base_score": candidate["base_score"],
        }

        if rank is not None:
            data["rank"] = rank
            data["adjusted_score"] = candidate.get("adjusted_score", candidate["base_score"])

        return data