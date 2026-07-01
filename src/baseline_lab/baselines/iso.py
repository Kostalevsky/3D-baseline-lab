from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont


@dataclass
class IsoResult:
    collage: Image.Image
    metadata: dict


class IsoOrthographicBaseline:
    """
    ISO baseline: строит ортографические виды 3D-объекта.

    Поддерживаются две схемы:
    - third-angle projection
    - first-angle projection

    На выходе:
    - коллаж с видами
    - metadata с технической информацией
    """

    def __init__(
        self,
        projection: str = "third",
        image_size: int = 512,
        background_color: tuple[int, int, int] = (255, 255, 255),
    ):
        if projection not in {"first", "third"}:
            raise ValueError("projection must be 'first' or 'third'")

        self.projection = projection
        self.image_size = image_size
        self.background_color = background_color

    def run(self, mesh: trimesh.Trimesh) -> IsoResult:
        """
        Запускает ISO baseline.

        Важно: renderer создаётся один раз на все виды.
        На macOS частое создание OffscreenRenderer может приводить
        к ошибкам pyglet / Cocoa / OpenGL.
        """
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_size,
            viewport_height=self.image_size,
        )

        try:
            views = {
                "front": self._render_view(mesh, renderer, eye=np.array([0, -2.5, 0])),
                "back": self._render_view(mesh, renderer, eye=np.array([0, 2.5, 0])),
                "right": self._render_view(mesh, renderer, eye=np.array([2.5, 0, 0])),
                "left": self._render_view(mesh, renderer, eye=np.array([-2.5, 0, 0])),
                "top": self._render_view(mesh, renderer, eye=np.array([0, 0, 2.5])),
                "bottom": self._render_view(mesh, renderer, eye=np.array([0, 0, -2.5])),
            }
        finally:
            renderer.delete()

        collage = self._make_collage(views)

        metadata = {
            "baseline": "iso_orthographic",
            "projection": self.projection,
            "views": list(views.keys()),
            "image_size": self.image_size,
        }

        return IsoResult(collage=collage, metadata=metadata)

    def save(self, result: IsoResult, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.collage.save(output_path)

    def _render_view(
        self,
        mesh: trimesh.Trimesh,
        renderer: pyrender.OffscreenRenderer,
        eye: np.ndarray,
    ) -> Image.Image:
        """
        Рендерит один ортографический вид объекта.

        eye — позиция камеры.
        Камера всегда смотрит в центр координат.
        """
        scene = pyrender.Scene(
            bg_color=[*self.background_color, 255],
            ambient_light=[0.4, 0.4, 0.4],
        )

        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.45, 0.45, 0.45, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.65,
        )

        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene.add(render_mesh)

        camera = pyrender.OrthographicCamera(xmag=0.8, ymag=0.8)

        camera_pose = self._look_at(
            eye=eye,
            target=np.array([0.0, 0.0, 0.0]),
            up=self._camera_up_vector(eye),
        )

        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
        scene.add(light, pose=camera_pose)

        color, _ = renderer.render(scene)
        return Image.fromarray(color)
    

    def _look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """
        Создаёт pose-матрицу камеры.

        Это стандартная операция:
        камера находится в eye и смотрит в target.
        """
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

    def _camera_up_vector(self, eye: np.ndarray) -> np.ndarray:
        """
        Выбирает направление 'вверх' для камеры.

        Для top/bottom видов нельзя использовать обычный up=(0,0,1),
        потому что камера смотрит вдоль Z.
        """
        if abs(eye[2]) > 0:
            return np.array([0.0, 1.0, 0.0])

        return np.array([0.0, 0.0, 1.0])

    def _make_collage(self, views: dict[str, Image.Image]) -> Image.Image:
        """
        Собирает ISO-коллаж.

        third-angle:
                TOP
        LEFT   FRONT   RIGHT   BACK
                BOTTOM

        first-angle:
                BOTTOM
        RIGHT  FRONT   LEFT    BACK
                TOP
        """
        cell = self.image_size
        label_height = 44
        gap = 16

        tile_width = cell
        tile_height = cell + label_height

        canvas_width = tile_width * 4 + gap * 3
        canvas_height = tile_height * 3 + gap * 2

        collage = Image.new(
            "RGB",
            (canvas_width, canvas_height),
            self.background_color,
        )

        if self.projection == "third":
            layout = {
                "top": (1, 0),
                "left": (0, 1),
                "front": (1, 1),
                "right": (2, 1),
                "back": (3, 1),
                "bottom": (1, 2),
            }
        else:
            layout = {
                "bottom": (1, 0),
                "right": (0, 1),
                "front": (1, 1),
                "left": (2, 1),
                "back": (3, 1),
                "top": (1, 2),
            }

        for name, image in views.items():
            col, row = layout[name]

            x = col * (tile_width + gap)
            y = row * (tile_height + gap)

            tile = self._add_label(image, name.upper())
            collage.paste(tile, (x, y))

        return collage


    def _add_label(self, image: Image.Image, label: str) -> Image.Image:
        """
        Добавляет подпись и рамку к каждому виду.
        """
        label_height = 44

        tile = Image.new(
            "RGB",
            (self.image_size, self.image_size + label_height),
            self.background_color,
        )

        tile.paste(image, (0, 0))

        draw = ImageDraw.Draw(tile)

        # Рамка вокруг изображения
        draw.rectangle(
            [0, 0, self.image_size - 1, self.image_size - 1],
            outline=(210, 210, 210),
            width=2,
        )

        # Шрифт: если системный не найдётся, Pillow возьмёт дефолтный
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        except OSError:
            font = ImageFont.load_default()

        text_position = (14, self.image_size + 10)
        draw.text(text_position, label, fill=(0, 0, 0), font=font)

        return tile