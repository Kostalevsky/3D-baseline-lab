from pathlib import Path

import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """
    Загружает 3D-модель из файла.

    Поддержка форматов зависит от trimesh:
    обычно работают .obj, .stl, .ply, .glb.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    loaded = trimesh.load(path, force="scene")

    # Иногда trimesh загружает не один mesh, а сцену с несколькими объектами.
    # Для baseline нам удобнее объединить всё в один mesh.
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for geometry in loaded.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        if not meshes:
            raise ValueError(f"No mesh geometry found in file: {path}")

        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported loaded object type: {type(loaded)}")

    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Нормализует модель:
    1. Переносит центр модели в начало координат.
    2. Масштабирует модель так, чтобы максимальный размер стал равен 1.

    Это нужно, чтобы разные модели рендерились сопоставимо.
    """
    mesh = mesh.copy()

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    mesh.apply_translation(-center)

    extents = mesh.extents
    max_extent = extents.max()

    if max_extent == 0:
        raise ValueError("Mesh has zero size and cannot be normalized.")

    mesh.apply_scale(1.0 / max_extent)

    return mesh