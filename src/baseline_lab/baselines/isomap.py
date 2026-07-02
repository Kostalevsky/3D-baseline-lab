from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler


@dataclass
class IsomapResult:
    collage: Image.Image
    metadata: dict
    extra_images: dict[str, Image.Image]


class IsomapViewSelectionBaseline:
    """
    Isomap View Selection baseline.

    Pipeline:
    1. Render 28 views.
    2. Extract visual features from every view.
    3. Project views into 2D Isomap embedding.
    4. Select diverse views from this embedding.
    5. Save all views + selected collages.
    """

    def __init__(
        self,
        image_size: int = 256,
        radius: float = 2.5,
        selected_counts: tuple[int, ...] = (3, 4, 6),
        background_color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.image_size = image_size
        self.radius = radius
        self.selected_counts = selected_counts
        self.background_color = background_color

    def run(self, mesh: trimesh.Trimesh) -> IsomapResult:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_size,
            viewport_height=self.image_size,
        )

        try:
            candidates = self._render_28_views(mesh, renderer)
        finally:
            renderer.delete()
        
        self._add_quality_scores(candidates)

        features = self._extract_features(candidates)
        embedding = self._compute_isomap_embedding(features)

        for candidate, point in zip(candidates, embedding):
            candidate["embedding_x"] = float(point[0])
            candidate["embedding_y"] = float(point[1])

        selected_by_count = {
            count: self._select_diverse_views_from_embedding(candidates, count)
            for count in self.selected_counts
        }

        all_views_collage = self._make_collage(
            candidates,
            cols=7,
            label_mode="view",
            title="ALL 28 VIEWS",
        )

        extra_images = {
            "all_views_collage.png": all_views_collage,
            "embedding.png": self._make_embedding_visualization(candidates, selected_by_count),
        }

        for count, selected in selected_by_count.items():
            extra_images[f"collage_{count}.png"] = self._make_collage(
                selected,
                cols=count,
                label_mode="selected",
                title=f"SELECTED {count} VIEWS",
            )

        # Main collage for compatibility with existing CLI.
        # We use collage_6 as the primary output.
        main_collage = extra_images["collage_6.png"]

        metadata = {
            "baseline": "isomap_view_selection",
            "description": "28-view rendering pipeline with Isomap-based view selection",
            "total_views": len(candidates),
            "selected_counts": list(self.selected_counts),
            "image_size": self.image_size,
            "radius": self.radius,
            "view_generation": {
                "elevation_15": "8 azimuth views",
                "elevation_35": "8 azimuth views",
                "elevation_60": "8 azimuth views",
                "elevation_80": "4 high-angle views",
                "total": "28 views",
            },
            "feature_extraction": {
                "image_features": "downsampled grayscale image",
                "mask_features": "area ratio, bbox fill score, edge ratio",
            },
            "selection": {
                "method": "quality-aware farthest point sampling in 2D Isomap embedding",
                "quality_threshold": 0.45,
                "selection_score": "0.60 * isomap_diversity + 0.40 * visual_quality",
                "outputs": [f"collage_{count}.png" for count in self.selected_counts],
            },
            "views": [
                self._candidate_metadata(candidate)
                for candidate in candidates
            ],
            "selected": {
                str(count): [
                    self._candidate_metadata(candidate)
                    for candidate in selected
                ]
                for count, selected in selected_by_count.items()
            },
            "extra_images": list(extra_images.keys()),
        }

        return IsomapResult(
            collage=main_collage,
            metadata=metadata,
            extra_images=extra_images,
        )

    def save(self, result: IsomapResult, output_path: str | Path) -> None:
        """
        Existing CLI passes output_path like:
        outputs/isomap/test_part_isomap/collage.png

        For this baseline we save several files into the same folder:
        - collage.png
        - all_views_collage.png
        - collage_3.png
        - collage_4.png
        - collage_6.png
        - embedding.png
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result.collage.save(output_path)

        for filename, image in result.extra_images.items():
            image.save(output_path.parent / filename)

    def _render_28_views(
        self,
        mesh: trimesh.Trimesh,
        renderer: pyrender.OffscreenRenderer,
    ) -> list[dict]:
        view_specs = []

        # 24 views: 8 azimuths × 3 elevations
        for elevation_deg in (15.0, 35.0, 60.0):
            for azimuth_deg in range(0, 360, 45):
                view_specs.append((float(azimuth_deg), elevation_deg))

        # 4 high-angle views
        for azimuth_deg in (0.0, 90.0, 180.0, 270.0):
            view_specs.append((azimuth_deg, 80.0))

        candidates = []

        for index, (azimuth_deg, elevation_deg) in enumerate(view_specs):
            eye = self._camera_position(
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
            )

            image = self._render_view(
                mesh=mesh,
                renderer=renderer,
                eye=eye,
            )

            score_data = self._compute_basic_image_scores(image)

            candidates.append(
                {
                    "index": index,
                    "name": f"view_{index:02d}_az{int(azimuth_deg):03d}_el{int(elevation_deg):02d}",
                    "azimuth_deg": azimuth_deg,
                    "elevation_deg": elevation_deg,
                    "eye": eye,
                    "image": image,
                    **score_data,
                }
            )

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

    def _compute_basic_image_scores(self, image: Image.Image) -> dict:
        arr = np.asarray(image.convert("RGB")).astype(np.float32)

        mask = np.any(arr < 245, axis=2)

        height, width = mask.shape
        image_area = height * width

        object_pixels = int(mask.sum())

        if object_pixels == 0:
            return {
                "area_ratio": 0.0,
                "bbox_fill_score": 0.0,
                "edge_ratio": 0.0,
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

        return {
            "area_ratio": float(area_ratio),
            "bbox_fill_score": float(bbox_fill_score),
            "edge_ratio": float(edge_ratio),
        }
    
    def _add_quality_scores(self, candidates: list[dict]) -> None:
        """
        Adds visual quality score to every candidate view.

        This score is not used instead of Isomap.
        It is used together with Isomap diversity during view selection.
        """
        area_values = np.array([candidate["area_ratio"] for candidate in candidates])
        edge_values = np.array([candidate["edge_ratio"] for candidate in candidates])
        bbox_values = np.array([candidate["bbox_fill_score"] for candidate in candidates])

        area_scores = self._normalize_values(area_values)
        edge_scores = self._normalize_values(edge_values)
        bbox_scores = self._normalize_values(bbox_values)

        for candidate, area_score, edge_score, bbox_score in zip(
            candidates,
            area_scores,
            edge_scores,
            bbox_scores,
        ):
            quality_score = (
                0.45 * area_score
                + 0.35 * edge_score
                + 0.20 * bbox_score
            )

            candidate["area_score"] = float(area_score)
            candidate["edge_score"] = float(edge_score)
            candidate["bbox_score"] = float(bbox_score)
            candidate["quality_score"] = float(quality_score)


    def _normalize_values(self, values: np.ndarray) -> np.ndarray:
        min_value = float(values.min())
        max_value = float(values.max())

        if max_value - min_value < 1e-9:
            return np.zeros_like(values, dtype=np.float32)

        return ((values - min_value) / (max_value - min_value)).astype(np.float32)

    def _extract_features(self, candidates: list[dict]) -> np.ndarray:
        feature_vectors = []

        for candidate in candidates:
            image = candidate["image"]

            # Low-resolution grayscale image as appearance feature.
            small = image.convert("L").resize((32, 32))
            image_vector = np.asarray(small).astype(np.float32).flatten() / 255.0

            scalar_features = np.array(
                [
                    candidate["area_ratio"],
                    candidate["bbox_fill_score"],
                    candidate["edge_ratio"],
                    candidate["azimuth_deg"] / 360.0,
                    candidate["elevation_deg"] / 90.0,
                ],
                dtype=np.float32,
            )

            feature_vector = np.concatenate([image_vector, scalar_features])
            feature_vectors.append(feature_vector)

        features = np.vstack(feature_vectors)

        # Standardize features before Isomap.
        features = StandardScaler().fit_transform(features)

        return features

    def _compute_isomap_embedding(self, features: np.ndarray) -> np.ndarray:
        n_samples = features.shape[0]

        # Keep n_neighbors safely smaller than n_samples.
        n_neighbors = min(6, n_samples - 1)

        isomap = Isomap(
            n_neighbors=n_neighbors,
            n_components=2,
        )

        embedding = isomap.fit_transform(features)

        return embedding

    def _select_diverse_views_from_embedding(
        self,
        candidates: list[dict],
        count: int,
    ) -> list[dict]:
        """
        Select diverse and informative views using quality-aware farthest point sampling.

        1. Filter out visually weak views.
        2. Start from the visually strongest view.
        3. Iteratively add views that are both:
        - far from already selected views in Isomap space;
        - visually informative.
        """
        if count >= len(candidates):
            return candidates

        quality_threshold = 0.45

        eligible_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate["quality_score"] >= quality_threshold
        ]

        # Fallback: if threshold is too strict for some object,
        # use all candidates instead of failing.
        if len(eligible_indices) < count:
            eligible_indices = list(range(len(candidates)))

        points = np.array(
            [
                [candidate["embedding_x"], candidate["embedding_y"]]
                for candidate in candidates
            ],
            dtype=np.float32,
        )

        quality_scores = np.array(
            [candidate["quality_score"] for candidate in candidates],
            dtype=np.float32,
        )

        # Start from the most informative eligible view.
        first_index = max(
            eligible_indices,
            key=lambda idx: quality_scores[idx],
        )

        selected_indices = [first_index]
        remaining_indices = set(eligible_indices) - {first_index}

        while len(selected_indices) < count and remaining_indices:
            selected_points = points[selected_indices]

            raw_distances = {}

            for idx in remaining_indices:
                point = points[idx]
                distances = np.linalg.norm(selected_points - point, axis=1)
                raw_distances[idx] = float(distances.min())

            max_distance = max(raw_distances.values())

            best_index = None
            best_score = -1.0

            for idx in remaining_indices:
                if max_distance < 1e-9:
                    diversity_score = 0.0
                else:
                    diversity_score = raw_distances[idx] / max_distance

                quality_score = quality_scores[idx]

                # Balance:
                # 60% — diversity in Isomap space
                # 40% — visual informativeness
                selection_score = 0.60 * diversity_score + 0.40 * quality_score

                if selection_score > best_score:
                    best_score = selection_score
                    best_index = idx

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        selected = []

        for rank, idx in enumerate(selected_indices):
            candidate = candidates[idx].copy()
            candidate["selection_rank"] = rank
            selected.append(candidate)

        return selected

    def _make_collage(
        self,
        candidates: list[dict],
        cols: int,
        label_mode: str,
        title: str | None = None,
    ) -> Image.Image:
        rows = math.ceil(len(candidates) / cols)

        label_height = 40
        title_height = 52 if title else 0
        gap = 14

        tile_width = self.image_size
        tile_height = self.image_size + label_height

        canvas_width = cols * tile_width + (cols - 1) * gap
        canvas_height = title_height + rows * tile_height + (rows - 1) * gap

        collage = Image.new(
            "RGB",
            (canvas_width, canvas_height),
            self.background_color,
        )

        draw = ImageDraw.Draw(collage)

        try:
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        except OSError:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        if title:
            draw.text((12, 14), title, fill=(0, 0, 0), font=title_font)

        for idx, candidate in enumerate(candidates):
            col = idx % cols
            row = idx // cols

            x = col * (tile_width + gap)
            y = title_height + row * (tile_height + gap)

            if label_mode == "selected":
                label = (
                    f"#{candidate.get('selection_rank', idx)} "
                    f"AZ{int(candidate['azimuth_deg']):03d} "
                    f"EL{int(candidate['elevation_deg']):02d}"
                )
            else:
                label = (
                    f"{candidate['index']:02d} "
                    f"AZ{int(candidate['azimuth_deg']):03d} "
                    f"EL{int(candidate['elevation_deg']):02d}"
                )

            tile = self._add_label(candidate["image"], label, label_font)
            collage.paste(tile, (x, y))

        return collage

    def _add_label(
        self,
        image: Image.Image,
        label: str,
        font: ImageFont.ImageFont,
    ) -> Image.Image:
        label_height = 40

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

        draw.text((10, self.image_size + 11), label, fill=(0, 0, 0), font=font)

        return tile

    def _make_embedding_visualization(
        self,
        candidates: list[dict],
        selected_by_count: dict[int, list[dict]],
    ) -> Image.Image:
        width = 900
        height = 700
        margin = 70

        image = Image.new("RGB", (width, height), self.background_color)
        draw = ImageDraw.Draw(image)

        try:
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
        except OSError:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        draw.text((24, 22), "ISOMAP VIEW EMBEDDING", fill=(0, 0, 0), font=title_font)

        xs = np.array([candidate["embedding_x"] for candidate in candidates])
        ys = np.array([candidate["embedding_y"] for candidate in candidates])

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        def scale_x(x: float) -> int:
            if x_max == x_min:
                return width // 2
            return int(margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin))

        def scale_y(y: float) -> int:
            if y_max == y_min:
                return height // 2
            return int(height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin))

        selected_6_indices = {
            candidate["index"]
            for candidate in selected_by_count.get(6, [])
        }

        for candidate in candidates:
            x = scale_x(candidate["embedding_x"])
            y = scale_y(candidate["embedding_y"])

            idx = candidate["index"]

            if idx in selected_6_indices:
                radius = 8
                fill = (0, 0, 0)
            else:
                radius = 5
                fill = (140, 140, 140)

            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=fill,
            )

            draw.text(
                (x + 9, y - 8),
                str(idx),
                fill=(0, 0, 0),
                font=label_font,
            )

        draw.text(
            (24, height - 38),
            "Black points = selected for collage_6",
            fill=(0, 0, 0),
            font=label_font,
        )

        return image

    def _candidate_metadata(self, candidate: dict) -> dict:
        data = {
            "index": candidate["index"],
            "name": candidate["name"],
            "azimuth_deg": candidate["azimuth_deg"],
            "elevation_deg": candidate["elevation_deg"],
            "area_ratio": candidate["area_ratio"],
            "bbox_fill_score": candidate["bbox_fill_score"],
            "edge_ratio": candidate["edge_ratio"],
            "edge_ratio": candidate["edge_ratio"],
            "quality_score": candidate.get("quality_score"),
            "area_score": candidate.get("area_score"),
            "edge_score": candidate.get("edge_score"),
            "bbox_score": candidate.get("bbox_score"),
            "embedding": [
                candidate.get("embedding_x"),
                candidate.get("embedding_y"),
            ],
        }

        if "selection_rank" in candidate:
            data["selection_rank"] = candidate["selection_rank"]

        return data