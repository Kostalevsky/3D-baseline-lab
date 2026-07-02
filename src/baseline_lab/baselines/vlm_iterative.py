from dataclasses import dataclass
from pathlib import Path
import base64
import io
import json
import os
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from dotenv import load_dotenv

from baseline_lab.baselines.isomap import IsomapViewSelectionBaseline


load_dotenv()

@dataclass
class VLMIterativeResult:
    collage: Image.Image
    metadata: dict
    extra_images: dict[str, Image.Image]


class VLMIterativeViewSelectionBaseline:
    """
    VLM-guided iterative view selection baseline.

    Pipeline:
    1. Render 28 candidate views.
    2. Create all-views collage.
    3. Create an initial selected collage.
    4. Ask a multimodal OpenAI model to improve the selected view set.
    5. Repeat for several iterations.
    6. Save final collage, iteration collages and metadata.
    """

    def __init__(
        self,
        image_size: int = 256,
        target_views: int = 6,
        iterations: int = 2,
        model: str | None = None,
        background_color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.image_size = image_size
        self.target_views = target_views
        self.iterations = iterations
        self.model = model or os.getenv("QWEN_VLM_MODEL", "qwen-vl-plus")
        self.background_color = background_color

        qwen_api_key = os.getenv("QWEN_API_KEY")
        qwen_base_url = os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

        if not qwen_api_key:
            raise RuntimeError(
                "QWEN_API_KEY is not set. Add it to a .env file or export it as an environment variable."
            )

        self.client = OpenAI(
            api_key=qwen_api_key,
            base_url=qwen_base_url,
        )

        # We reuse the already implemented 28-view rendering and quality scoring
        # from the Isomap baseline to avoid duplicating camera/rendering logic.
        self.view_generator = IsomapViewSelectionBaseline(
            image_size=image_size,
            selected_counts=(target_views,),
            background_color=background_color,
        )

    def run(self, mesh: trimesh.Trimesh) -> VLMIterativeResult:
        candidates = self._generate_candidates(mesh)

        all_views_collage = self._make_collage(
            candidates,
            cols=7,
            title="ALL 28 CANDIDATE VIEWS",
            label_mode="candidate",
        )

        current_indices = self._initial_selection(candidates)

        extra_images: dict[str, Image.Image] = {
            "all_views_collage.png": all_views_collage,
        }

        initial_selected = [candidates[idx] for idx in current_indices]
        extra_images["iteration_0_collage.png"] = self._make_collage(
            initial_selected,
            cols=self.target_views,
            title="ITERATION 0 — INITIAL SELECTION",
            label_mode="selected",
        )

        iteration_logs = []

        for iteration in range(1, self.iterations + 1):
            current_selected = [candidates[idx] for idx in current_indices]

            current_collage = self._make_collage(
                current_selected,
                cols=self.target_views,
                title=f"ITERATION {iteration - 1} — CURRENT SELECTION",
                label_mode="selected",
            )

            decision = self._ask_vlm_to_select_views(
                candidates=candidates,
                all_views_collage=all_views_collage,
                current_collage=current_collage,
                current_indices=current_indices,
                iteration=iteration,
            )

            proposed_indices = decision.get("selected_view_indices", [])

            validated_indices = self._validate_selected_indices(
                proposed_indices=proposed_indices,
                fallback_indices=current_indices,
                total_candidates=len(candidates),
            )

            current_indices = validated_indices

            selected_candidates = [candidates[idx] for idx in current_indices]

            iteration_collage = self._make_collage(
                selected_candidates,
                cols=self.target_views,
                title=f"ITERATION {iteration} — VLM SELECTION",
                label_mode="selected",
            )

            extra_images[f"iteration_{iteration}_collage.png"] = iteration_collage

            iteration_logs.append(
                {
                    "iteration": iteration,
                    "input_selected_indices": decision.get(
                        "previous_selected_view_indices",
                        current_indices,
                    ),
                    "selected_view_indices": current_indices,
                    "vlm_decision": decision,
                }
            )

        final_candidates = [candidates[idx] for idx in current_indices]

        final_collage = self._make_collage(
            final_candidates,
            cols=self.target_views,
            title="FINAL VLM-GUIDED SELECTION",
            label_mode="selected",
        )

        extra_images["final_collage.png"] = final_collage

        metadata = {
            "baseline": "vlm_iterative_view_selection",
            "description": "VLM-guided iterative selection of informative 3D object views",
            "model": self.model,
            "image_size": self.image_size,
            "candidate_views": len(candidates),
            "target_views": self.target_views,
            "iterations": self.iterations,
            "view_generation": {
                "source": "same 28-view candidate setup as Isomap View Selection baseline",
                "elevation_15": "8 azimuth views",
                "elevation_35": "8 azimuth views",
                "elevation_60": "8 azimuth views",
                "elevation_80": "4 high-angle views",
                "total": "28 views",
            },
            "initial_selection": {
                "method": "top visual quality scores from 28 candidates",
                "selected_view_indices": self._initial_selection(candidates),
            },
            "final_selection": {
                "selected_view_indices": current_indices,
                "selected_views": [
                    self._candidate_metadata(candidate)
                    for candidate in final_candidates
                ],
            },
            "iterations_log": iteration_logs,
            "all_candidates": [
                self._candidate_metadata(candidate)
                for candidate in candidates
            ],
            "extra_images": list(extra_images.keys()),
        }

        return VLMIterativeResult(
            collage=final_collage,
            metadata=metadata,
            extra_images=extra_images,
        )

    def save(self, result: VLMIterativeResult, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Main output for compatibility with existing CLI.
        result.collage.save(output_path)

        for filename, image in result.extra_images.items():
            image.save(output_path.parent / filename)

    def _generate_candidates(self, mesh: trimesh.Trimesh) -> list[dict]:
        """
        Reuses rendering logic from Isomap baseline:
        - render 28 candidate views;
        - calculate visual quality scores.
        """
        import pyrender

        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_size,
            viewport_height=self.image_size,
        )

        try:
            candidates = self.view_generator._render_28_views(mesh, renderer)
        finally:
            renderer.delete()

        self.view_generator._add_quality_scores(candidates)

        return candidates

    def _initial_selection(self, candidates: list[dict]) -> list[int]:
        """
        Initial selection is deterministic:
        choose top-N views by visual quality, while avoiding exact duplicates.
        """
        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: candidate["quality_score"],
            reverse=True,
        )

        selected = []

        for candidate in sorted_candidates:
            idx = candidate["index"]

            if len(selected) >= self.target_views:
                break

            selected.append(idx)

        return selected

    def _ask_vlm_to_select_views(
        self,
        candidates: list[dict],
        all_views_collage: Image.Image,
        current_collage: Image.Image,
        current_indices: list[int],
        iteration: int,
    ) -> dict:
        prompt = self._build_prompt(
            candidates=candidates,
            current_indices=current_indices,
            iteration=iteration,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_data_url(all_views_collage),
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_data_url(current_collage),
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw_text = response.choices[0].message.content

        if raw_text is None:
            raise RuntimeError("Qwen response is empty.")

        try:
            decision = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Qwen response is not valid JSON: {raw_text}"
            ) from exc

        return decision

    def _build_prompt(
        self,
        candidates: list[dict],
        current_indices: list[int],
        iteration: int,
    ) -> str:
        candidate_table = self._build_candidate_table(candidates)

        return f"""
        You are selecting the most informative rendered views of a 3D object for a visual collage.

        The object is a 3D mesh rendered from 28 candidate camera views.
        The first image shows all 28 candidate views with labels.
        The second image shows the currently selected collage.

        Your task:
        Select exactly {self.target_views} view indices from 0 to 27.

        Selection goal:
        Choose a compact set of views that helps a multimodal model understand:
        - overall object shape;
        - holes and cutouts;
        - silhouette;
        - thickness;
        - symmetry/asymmetry;
        - important side or diagonal information;
        - spatial structure.

        Important rules:
        1. Return exactly {self.target_views} indices.
        2. Use only candidate indices from 0 to 27.
        3. Avoid visually redundant views.
        4. Avoid weak side views where the object looks like a thin strip, unless they add important thickness information.
        5. Prefer views that show holes, contour, silhouette, thickness and global shape.
        6. Balance visual informativeness and diversity.
        7. Do not invent views that are not in the candidate list.
        8. Keep useful current views if they are already informative.
        9. Replace current views if they are redundant or visually weak.

        Current iteration: {iteration}

        Currently selected indices:
        {current_indices}

        Candidate view metadata:
        {candidate_table}

        Return JSON with exactly these fields:
        {{
            "previous_selected_view_indices": [integer],
            "selected_view_indices": [integer],
            "removed_view_indices": [integer],
            "added_view_indices": [integer],
            "reasoning": "string",
            "visible_features": ["string"],
            "missing_or_weak_features": ["string"],
            "confidence": number
        }}

        Return only valid JSON. Do not include markdown, comments, explanations, or code fences.

        Return only valid JSON. Do not include markdown, comments, explanations, or code fences.
        """.strip()

    def _build_candidate_table(self, candidates: list[dict]) -> str:
        rows = []

        for candidate in candidates:
            rows.append(
                (
                    f"index={candidate['index']}, "
                    f"azimuth={candidate['azimuth_deg']:.0f}, "
                    f"elevation={candidate['elevation_deg']:.0f}, "
                    f"quality={candidate['quality_score']:.3f}, "
                    f"area={candidate['area_ratio']:.3f}, "
                    f"edge={candidate['edge_ratio']:.3f}, "
                    f"bbox_fill={candidate['bbox_fill_score']:.3f}"
                )
            )

        return "\n".join(rows)

    def _validate_selected_indices(
        self,
        proposed_indices: list[Any],
        fallback_indices: list[int],
        total_candidates: int,
    ) -> list[int]:
        valid = []

        for idx in proposed_indices:
            if not isinstance(idx, int):
                continue

            if idx < 0 or idx >= total_candidates:
                continue

            if idx not in valid:
                valid.append(idx)

        if len(valid) == self.target_views:
            return valid

        # If the model returns too few valid views, complete with fallback.
        for idx in fallback_indices:
            if idx not in valid:
                valid.append(idx)
            if len(valid) == self.target_views:
                return valid

        # Last fallback: fill with first valid indices.
        for idx in range(total_candidates):
            if idx not in valid:
                valid.append(idx)
            if len(valid) == self.target_views:
                return valid

        return valid[: self.target_views]

    def _image_to_data_url(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _make_collage(
        self,
        candidates: list[dict],
        cols: int,
        title: str,
        label_mode: str,
    ) -> Image.Image:
        rows = int(np.ceil(len(candidates) / cols))

        label_height = 42
        title_height = 54
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

        draw.text((12, 14), title, fill=(0, 0, 0), font=title_font)

        for idx, candidate in enumerate(candidates):
            col = idx % cols
            row = idx // cols

            x = col * (tile_width + gap)
            y = title_height + row * (tile_height + gap)

            if label_mode == "selected":
                label = (
                    f"{candidate['index']:02d} "
                    f"AZ{int(candidate['azimuth_deg']):03d} "
                    f"EL{int(candidate['elevation_deg']):02d} "
                    f"Q{candidate.get('quality_score', 0):.2f}"
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

        draw.text(
            (10, self.image_size + 12),
            label,
            fill=(0, 0, 0),
            font=font,
        )

        return tile

    def _candidate_metadata(self, candidate: dict) -> dict:
        return {
            "index": candidate["index"],
            "name": candidate["name"],
            "azimuth_deg": candidate["azimuth_deg"],
            "elevation_deg": candidate["elevation_deg"],
            "area_ratio": candidate["area_ratio"],
            "bbox_fill_score": candidate["bbox_fill_score"],
            "edge_ratio": candidate["edge_ratio"],
            "quality_score": candidate.get("quality_score"),
            "area_score": candidate.get("area_score"),
            "edge_score": candidate.get("edge_score"),
            "bbox_score": candidate.get("bbox_score"),
        }